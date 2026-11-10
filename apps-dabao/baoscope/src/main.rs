//! BaoScope v0 - BIO-powered logic analyzer for Dabao.
//!
//! * BIO core A ("sampler") waits for its quantum clock, then copies the GPIO
//!   input register (x21) into FIFO1 (x17). One 32-bit word per sample.
//! * BIO core B ("generator") drives a 4-bit binary counter on PC0..PC3
//!   (BIO bits 16..19), giving a known self-test signal with no wiring.
//! * The main CPU drains FIFO1 into RAM, packs the 8 channel bits into one
//!   byte per sample, and ships the capture over USB CDC.
//!
//! Host -> device: text lines, '\n' terminated:
//!   ping                        -> PONG message
//!   cap <rate_hz> <n> [gen_hz]  -> CAP1 message (gen_hz = 0 disables the generator)
//!   stream <batches>            -> <batches> STRM messages of filler (throughput test)
//!
//! Device -> host: a stream of fixed 512-byte records, so the host can detect
//! duplicated or lost USB packets (the bao1x USB CDC path has been observed to
//! re-send full 512-byte packets):
//!   record  = 0xB5 0x5C | seq u16 | 504 payload bytes | crc32(previous 508 bytes) u32
//!   message = kind [u8;4] | len u32 | body[len], split across consecutive record
//!             payloads, zero-padded to a whole record
//!   CAP1 body = version u32 | rate_hz u32 | n u32 | elapsed_ms u32 | full_hits u32
//!               | n sample bytes (bit i = channel i)
//!   ERR_ body = UTF-8 error text
//!
//! All integers little-endian.

use bao1x_api::bio::*;
use bao1x_api::bio_code;
use bao1x_api::bio_resources::*;
use bao1x_hal::bio::{Bio, CoreCsr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use usb_bao1x::{UsbDeviceState, UsbHid};
use utralib::utra::bio_bdma;

/// Real (non-BIO) traffic sources for testing the decoders against actual chip peripherals.
/// Each runs on its own thread until `stop` is set.
enum RealTraffic {
    /// Numbered log lines; the log server sends them out of UART2 TX (PB14, 1 Mbaud 8N1).
    UartLog,
    /// Repeated address scan 0x08..=0x77 on the hardware I2C0 controller (PB11 SCL, PB12 SDA).
    I2cScan,
}

fn start_traffic(kind: RealTraffic, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<u32> {
    std::thread::spawn(move || {
        let mut count = 0u32;
        match kind {
            RealTraffic::UartLog => {
                let t = ticktimer::Ticktimer::new().unwrap();
                while !stop.load(Ordering::Relaxed) {
                    log::info!("BSCP-REAL {:06}", count);
                    count += 1;
                    t.sleep_ms(1).ok();
                }
            }
            RealTraffic::I2cScan => {
                use bao1x_api::I2cApi;
                let mut i2c = bao1x_hal::i2c::I2c::new();
                let mut buf = [0u8; 1];
                'outer: loop {
                    for addr in 0x08u8..=0x77 {
                        if stop.load(Ordering::Relaxed) {
                            break 'outer;
                        }
                        // NACK or bus error are both fine here: we only need real bus activity.
                        let _ = i2c.i2c_read(addr, 0x00, &mut buf, false);
                        count += 1;
                    }
                }
            }
        }
        count
    })
}

const PROTOCOL_VERSION: u32 = 10;
/// BIO bit numbers sampled as channels 0..7. Bit n = PB(n) for n<16, PC(n-16) otherwise.
/// These are all in dabao-console's ALLOWED_PINS list (free, not tied to board functions).
const CH_BITS: [u32; 8] = [16, 17, 18, 19, 23, 24, 26, 27];
/// Pins the decoders may read in place WITHOUT claiming or remapping them: the chip's own
/// UART2 TX (PB14 = BIO14) and I2C0 SCL/SDA (PB11/PB12 = BIO11/12). The hardware peripheral
/// keeps driving the pad; BIO only observes it via x21 - wire-free sniffing of on-chip buses.
const SNIFF_BITS: [u32; 3] = [11, 12, 14];

fn decodable(bit: u32) -> bool { CH_BITS.contains(&bit) || SNIFF_BITS.contains(&bit) }

const MAX_SAMPLES: usize = 65536;
/// Spin-loop iterations to wait for a sample before declaring the BIO stuck.
const STALL_SPINS: u32 = 20_000_000;

/// DMA capture: the BIO core stores one u32 per sample straight into RAM.
const PAGE: usize = 4096;
const WORDS_PER_PAGE: usize = PAGE / 4;
const DMA_PAGES: usize = 64; // 256 KiB -> 65536 samples
/// Max samples in 32-bit mode; packed modes multiply this by samples-per-word.
const DMA_MAX_SAMPLES: usize = DMA_PAGES * WORDS_PER_PAGE;
/// Cap on samples returned in one CAP1 message (bounded by app heap for the reply buffers).
const MAX_OUT_SAMPLES: usize = 131072;
/// Byte offset in the sampler core's 4 KiB IMEM where the page table lives
/// (word 0 = page count, then one physical page address per word).
const PAGE_TABLE_OFFSET: usize = 0x400;

struct Scope {
    bio: Bio,
    grant: ResourceGrant,
    _handle: CoreHandle,
    fifo: CoreCsr,
    _handle2: CoreHandle,
    fifo2: CoreCsr,
    /// Set up lazily on the first `dcap`, so a DMA setup failure is reported over USB
    /// instead of killing the app before it can answer `ping`.
    dma_buf: Option<xous::MemoryRange>,
    /// Physical address of each 4 KiB page of `dma_buf` (pages need not be contiguous).
    dma_pages: Vec<u32>,
}

impl Resources for Scope {
    fn resource_spec() -> ResourceSpec {
        ResourceSpec {
            claimer: "baoscope".to_string(),
            // cores[0] = sampler, cores[1] = DMA writer (split mode), cores[2] = test-signal generator
            cores: vec![CoreRequirement::Any, CoreRequirement::Any, CoreRequirement::Any],
            // FIFO1 = results/done to the CPU, FIFO2 = sampler -> writer data path
            fifos: vec![Fifo::Fifo1, Fifo::Fifo2],
            static_pins: CH_BITS.iter().map(|&b| b as u8).collect(),
            dynamic_pin_count: 0,
        }
    }
}

impl Scope {
    fn new() -> Result<Self, BioError> {
        let mut bio = Bio::new();
        let spec = Self::resource_spec();
        let grant = bio.claim_resources(&spec)?;
        let io = IoConfig::from_resource_spec(&spec, &Some(IoConfigMode::SetOnly)).unwrap();
        bio.setup_io_config(io)?;
        // safety: the handle is stored in Self alongside the CSR view, so it outlives it
        let handle = unsafe { bio.get_core_handle(Fifo::Fifo1) }?.expect("no FIFO1 handle");
        let fifo = CoreCsr::from_handle(&handle);
        let handle2 = unsafe { bio.get_core_handle(Fifo::Fifo2) }?.expect("no FIFO2 handle");
        let fifo2 = CoreCsr::from_handle(&handle2);
        log::info!("baoscope: grant {:?}, BIO clock {} Hz", grant, bio.get_bio_freq());
        Ok(Self {
            bio,
            grant,
            _handle: handle,
            fifo,
            _handle2: handle2,
            fifo2,
            dma_buf: None,
            dma_pages: Vec::new(),
        })
    }

    /// Allocate the capture buffer, resolve its physical pages and open the DMA window.
    /// Every step reports a distinct error so failures are diagnosable over USB.
    fn ensure_dma(&mut self) -> Result<(), String> {
        if self.dma_buf.is_some() {
            return Ok(());
        }
        // RESERVE so every page is backed now and its physical address is stable.
        let mut buf = xous::syscall::map_memory(
            None,
            None,
            DMA_PAGES * PAGE,
            xous::MemoryFlags::R | xous::MemoryFlags::W | xous::MemoryFlags::RESERVE,
        )
        .map_err(|e| format!("dma step 1 map_memory({} KiB): {:?}", DMA_PAGES * 4, e))?;
        let base = buf.as_mut_ptr() as usize;
        // RESERVE alone did not back the pages on this kernel (virt_to_phys -> BadAddress on
        // page 0), so touch each page to fault it in before asking for its physical address.
        for p in 0..DMA_PAGES {
            unsafe { ((base + p * PAGE) as *mut u32).write_volatile(0) };
        }
        let mut pages = Vec::with_capacity(DMA_PAGES);
        for p in 0..DMA_PAGES {
            let phys = xous::syscall::virt_to_phys(base + p * PAGE)
                .map_err(|e| format!("dma step 2 virt_to_phys(page {}): {:?}", p, e))?;
            pages.push(phys as u32);
        }
        log::info!("DMA buffer virt {:x}, first phys pages {:x?}", base, &pages[..4]);

        // Let BIO bus masters write to SRAM. (Window 0 = all of SRAM; a tighter set of windows
        // around just the capture pages would be better hygiene - see PLAN.md.)
        let mut windows = DmaFilterWindows { windows: [None; 4] };
        windows.windows[0] = Some(DmaWindow {
            base: (utralib::generated::HW_SRAM_MEM as u32) >> 12,
            bounds: core::num::NonZeroU32::new((utralib::generated::HW_SRAM_MEM_LEN as u32) >> 12).unwrap(),
        });
        self.bio.setup_dma_windows(windows).map_err(|e| format!("dma step 3 setup_dma_windows: {:?}", e))?;
        self.dma_buf = Some(buf);
        self.dma_pages = pages;
        Ok(())
    }

    /// DMA capture: no CPU in the sample loop. Returns (sample bytes, rate, BIO cycles).
    /// `mode` = samples packed per 32-bit word by the BIO core:
    ///   1: raw GPIO word, 2: GPIO[31:16] twice, 8: GPIO[19:16] (ch0..3) eight times.
    fn capture_dma(
        &mut self,
        rate: u32,
        n: usize,
        gen_hz: u32,
        mode: usize,
        split: bool,
        timer: &ticktimer::Ticktimer,
    ) -> Result<(Vec<u8>, u32, u32), String> {
        let sampler = self.grant.cores[0];
        let writer = self.grant.cores[1];
        let generator = self.grant.cores[2];
        self.ensure_dma()?;
        let words = n.div_ceil(mode);
        let pages = words.div_ceil(WORDS_PER_PAGE).clamp(1, DMA_PAGES);
        let words = pages * WORDS_PER_PAGE; // the core always fills whole pages
        let n = n.min(words * mode);

        // Direct: the sampler stores to RAM itself. Split: the sampler only packs and pushes
        // words into FIFO2; the writer core pops them and stores, hiding store latency.
        let (sampler_code, ram_code) = if split {
            let s = match mode {
                2 => split_sampler2_code(),
                8 => split_sampler8_code(),
                _ => split_sampler1_code(),
            };
            (s.0, writer_code().0)
        } else {
            let s = match mode {
                2 => dma_sampler2_code(),
                8 => dma_sampler8_code(),
                _ => dma_sampler_code(),
            };
            (s.0, s.0)
        };
        // Whichever core walks the page table gets image = code, padding, table at 0x400.
        if ram_code.len() > PAGE_TABLE_OFFSET {
            return Err("DMA code overlaps page table".into());
        }
        let mut image = vec![0u8; PAGE_TABLE_OFFSET];
        image[..ram_code.len()].copy_from_slice(ram_code);
        image.extend_from_slice(&(pages as u32).to_le_bytes());
        for &p in &self.dma_pages[..pages] {
            image.extend_from_slice(&p.to_le_bytes());
        }

        // Poison the buffer so stale data can't masquerade as a capture.
        let buf = self.dma_buf.as_mut().unwrap().as_mut_ptr() as *mut u32;
        for i in 0..words {
            unsafe { buf.add(i).write_volatile(0xDEAD_BEEF) };
        }
        bao1x_hal::cache_flush();

        let quantum = CoreConfig { clock_mode: ClockMode::TargetFreqInt(rate) };
        let free_run = CoreConfig { clock_mode: ClockMode::FixedDivider(0, 0) };
        let actual = if split {
            self.bio.init_core(writer, (&image, None), free_run).map_err(|e| format!("init writer: {:?}", e))?;
            self.bio.init_core(sampler, (sampler_code, None), quantum)
        } else {
            self.bio.init_core(writer, idle_code(), free_run).map_err(|e| format!("init idle: {:?}", e))?;
            self.bio.init_core(sampler, (&image, None), quantum)
        }
        .map_err(|e| format!("init dma sampler: {:?}", e))?
        .unwrap_or(rate);
        let gen_code = if gen_hz > 0 { generator_code() } else { idle_code() };
        self.bio
            .init_core(generator, gen_code, CoreConfig { clock_mode: ClockMode::TargetFreqInt(gen_hz.max(1)) })
            .map_err(|e| format!("init generator: {:?}", e))?;
        let gen_ok = true;
        while self.fifo.csr.rf(bio_bdma::SFR_FLEVEL_PCLK_REGFIFO_LEVEL1) != 0 {
            let _ = self.fifo.csr.r(bio_bdma::SFR_RXF1);
        }
        while self.fifo2.csr.rf(bio_bdma::SFR_FLEVEL_PCLK_REGFIFO_LEVEL2) != 0 {
            let _ = self.fifo2.csr.r(bio_bdma::SFR_RXF2);
        }

        self.bio.set_core_run_state(&self.grant, true);
        // The core that writes RAM pushes its elapsed cycle count into FIFO1 when done.
        let budget_ms = ((words * mode) as u64 * 1000 / actual.max(1) as u64) * 2 + 500;
        let t0 = timer.elapsed_ms();
        let mut cycles = None;
        while timer.elapsed_ms() - t0 < budget_ms {
            if self.fifo.csr.rf(bio_bdma::SFR_FLEVEL_PCLK_REGFIFO_LEVEL1) != 0 {
                cycles = Some(self.fifo.csr.r(bio_bdma::SFR_RXF1));
                break;
            }
            timer.sleep_ms(1).ok();
        }
        self.bio.set_core_run_state(&self.grant, false);
        self.bio.de_init_core(sampler).ok();
        self.bio.de_init_core(writer).ok();
        if gen_ok {
            self.bio.de_init_core(generator).ok();
        }
        let cycles = cycles.ok_or_else(|| format!("DMA sampler did not finish within {} ms", budget_ms))?;

        bao1x_hal::cache_flush();
        let mut out = Vec::with_capacity(n);
        let mut untouched = 0usize;
        for i in 0..n.div_ceil(mode) {
            let w = unsafe { buf.add(i).read_volatile() };
            if w == 0xDEAD_BEEF {
                untouched += 1;
            }
            match mode {
                // low half = earlier sample (GPIO[31:16] shifted down), high half = later sample
                2 => {
                    out.push(pack(w << 16));
                    out.push(pack(w & 0xFFFF_0000));
                }
                // shift-register packing: earliest sample ends up in the lowest nibble
                8 => {
                    for k in 0..8 {
                        out.push(((w >> (4 * k)) & 0xF) as u8);
                    }
                }
                _ => out.push(pack(w)),
            }
        }
        out.truncate(n);
        if untouched > 0 {
            log::warn!("{} DMA words still hold the poison value", untouched);
            return Err(format!("{} DMA words never written (poison still present)", untouched));
        }
        Ok((out, actual, cycles))
    }

    /// On-chip UART decode (8N1, LSB first). cores[0] = decoder (quantum = 8 x baud),
    /// cores[1] = writer (FIFO2 -> RAM), cores[2] = optional UART test transmitter on the same pin.
    /// Each decoded byte is stored as one u32: bits[7:0] = data, bit 8 = framing error.
    /// Returns (words, actual decoder quantum Hz, writer finished?) - on timeout, the words
    /// written so far are returned (the buffer is poisoned, so unwritten words are detectable).
    fn decode_uart(
        &mut self,
        baud: u32,
        pin: u32,
        nbytes: usize,
        timeout_ms: u64,
        selftest: bool,
        timer: &ticktimer::Ticktimer,
    ) -> Result<(Vec<u32>, u32, bool), String> {
        let (words, actual, finished) = self.decode_run(
            &[pin],
            uart_rx_code().0,
            baud * 8,
            uart_tx_code().0,
            baud,
            nbytes,
            timeout_ms,
            selftest,
            timer,
        )?;
        Ok((words, actual, finished))
    }

    /// On-chip I2C decode. cores[0] = decoder (state machine on SCL=bit `scl`, SDA=bit `scl`+1,
    /// polled every quantum at `sample_hz`), cores[1] = writer, cores[2] = optional I2C master
    /// test source running its bit phases at 4 x `bus_hz`.
    /// Event word: tag<<16 | ack<<8 | data; tag 1 = START, 2 = STOP, 3 = BYTE (ack 0 = ACK).
    fn decode_i2c(
        &mut self,
        sample_hz: u32,
        bus_hz: u32,
        scl: u32,
        nevents: usize,
        timeout_ms: u64,
        selftest: bool,
        timer: &ticktimer::Ticktimer,
    ) -> Result<(Vec<u32>, u32, bool), String> {
        if !decodable(scl + 1) {
            return Err(format!("SDA bit {} (SCL+1) is not a claimed channel bit {:?}", scl + 1, CH_BITS));
        }
        self.decode_run(
            &[scl],
            i2c_rx_code().0,
            sample_hz,
            i2c_tx_code().0,
            bus_hz * 4,
            nevents,
            timeout_ms,
            selftest,
            timer,
        )
    }

    /// On-chip SPI decode (mode 0). Pins base..base+3 = SCK, MOSI, MISO, CS (default PC0..PC3).
    /// Event word: tag<<16 | miso<<8 | mosi; tag 1 = CS assert, 2 = CS release (low bits =
    /// leftover bit count), 3 = BYTE. Test source runs at 2 x `sck_hz` (one quantum per half-bit).
    fn decode_spi(
        &mut self,
        sample_hz: u32,
        sck_hz: u32,
        base: u32,
        nevents: usize,
        timeout_ms: u64,
        selftest: bool,
        timer: &ticktimer::Ticktimer,
    ) -> Result<(Vec<u32>, u32, bool), String> {
        if let Some(b) = (base..base + 4).find(|b| !CH_BITS.contains(b)) {
            return Err(format!("SPI pin bit {} (base+0..3) is not a claimed channel bit {:?}", b, CH_BITS));
        }
        self.decode_run(
            &[base],
            spi_rx_code().0,
            sample_hz,
            spi_tx_code().0,
            sck_hz * 2,
            nevents,
            timeout_ms,
            selftest,
            timer,
        )
    }

    /// Shared runner for on-chip decoders: decoder core -> FIFO2 -> writer core -> RAM, with an
    /// optional test-source core. `pins` is placed at IMEM 0x400 of both decoder and source.
    /// Returns (event words written, actual decoder quantum Hz, writer finished?).
    fn decode_run(
        &mut self,
        pins: &[u32],
        rx_code: &[u8],
        rx_hz: u32,
        tx_code: &[u8],
        tx_hz: u32,
        nbytes: usize,
        timeout_ms: u64,
        selftest: bool,
        timer: &ticktimer::Ticktimer,
    ) -> Result<(Vec<u32>, u32, bool), String> {
        if let Some(p) = pins.iter().find(|&&p| !decodable(p)) {
            return Err(format!("pin {} is not one of the claimed channel bits {:?}", p, CH_BITS));
        }
        // The test source drives its pins; never let it touch the chip's own peripheral pads.
        let touches_sniff = pins.iter().any(|&p| SNIFF_BITS.contains(&p) || SNIFF_BITS.contains(&(p + 1)));
        if selftest && touches_sniff {
            return Err("self-test is not allowed on in-place sniff pins (PB11/PB12/PB14)".into());
        }
        let decoder = self.grant.cores[0];
        let writer = self.grant.cores[1];
        let generator = self.grant.cores[2];
        self.ensure_dma()?;
        let pages = nbytes.div_ceil(WORDS_PER_PAGE).clamp(1, DMA_PAGES);
        let words = pages * WORDS_PER_PAGE;

        let mut table = vec![pages as u32];
        table.extend_from_slice(&self.dma_pages[..pages]);
        let writer_image = with_table(writer_code().0, &table)?;
        let decoder_image = with_table(rx_code, pins)?;
        let tx_image = with_table(tx_code, pins)?;

        let buf = self.dma_buf.as_mut().unwrap().as_mut_ptr() as *mut u32;
        for i in 0..words {
            unsafe { buf.add(i).write_volatile(0xDEAD_BEEF) };
        }
        bao1x_hal::cache_flush();

        let free_run = CoreConfig { clock_mode: ClockMode::FixedDivider(0, 0) };
        self.bio.init_core(writer, (&writer_image, None), free_run).map_err(|e| format!("init writer: {:?}", e))?;
        let actual = self
            .bio
            .init_core(decoder, (&decoder_image, None), CoreConfig { clock_mode: ClockMode::TargetFreqInt(rx_hz) })
            .map_err(|e| format!("init decoder: {:?}", e))?
            .unwrap_or(rx_hz);
        if selftest {
            self.bio
                .init_core(generator, (&tx_image, None), CoreConfig { clock_mode: ClockMode::TargetFreqInt(tx_hz) })
                .map_err(|e| format!("init test source: {:?}", e))?;
        } else {
            // Decoding an external signal: first force every BaoScope pin back to an input, so a
            // pin left driven by an earlier self-test can never fight the external driver.
            self.bio
                .init_core(generator, inputs_code(), free_run)
                .map_err(|e| format!("init pins-to-inputs: {:?}", e))?;
        }
        while self.fifo.csr.rf(bio_bdma::SFR_FLEVEL_PCLK_REGFIFO_LEVEL1) != 0 {
            let _ = self.fifo.csr.r(bio_bdma::SFR_RXF1);
        }
        while self.fifo2.csr.rf(bio_bdma::SFR_FLEVEL_PCLK_REGFIFO_LEVEL2) != 0 {
            let _ = self.fifo2.csr.r(bio_bdma::SFR_RXF2);
        }

        self.bio.set_core_run_state(&self.grant, true);
        let t0 = timer.elapsed_ms();
        let mut finished = false;
        while timer.elapsed_ms() - t0 < timeout_ms {
            if self.fifo.csr.rf(bio_bdma::SFR_FLEVEL_PCLK_REGFIFO_LEVEL1) != 0 {
                let _cycles = self.fifo.csr.r(bio_bdma::SFR_RXF1);
                finished = true;
                break;
            }
            timer.sleep_ms(1).ok();
        }
        self.bio.set_core_run_state(&self.grant, false);
        for core in [decoder, writer, generator] {
            self.bio.de_init_core(core).ok();
        }

        bao1x_hal::cache_flush();
        let mut out = Vec::new();
        for i in 0..words.min(nbytes) {
            let w = unsafe { buf.add(i).read_volatile() };
            if w == 0xDEAD_BEEF {
                break; // the writer stores in order, so the first poison word ends the data
            }
            out.push(w);
        }
        Ok((out, actual, finished))
    }

    /// Returns (raw samples, actual sample rate, full_hits) or an error string.
    fn capture(&mut self, rate: u32, n: usize, gen_hz: u32) -> Result<(Vec<u8>, u32, u32), String> {
        let sampler = self.grant.cores[0];
        let unused = self.grant.cores[1];
        let generator = self.grant.cores[2];

        // Every granted core gets started below, so park the unused one in an idle loop.
        self.bio
            .init_core(unused, idle_code(), CoreConfig { clock_mode: ClockMode::FixedDivider(0, 0) })
            .map_err(|e| format!("init idle: {:?}", e))?;
        let actual = self
            .bio
            .init_core(sampler, sampler_code(), CoreConfig { clock_mode: ClockMode::TargetFreqInt(rate) })
            .map_err(|e| format!("init sampler: {:?}", e))?
            .unwrap_or(rate);
        let gen_ok = if gen_hz > 0 {
            self.bio
                .init_core(generator, generator_code(), CoreConfig {
                    clock_mode: ClockMode::TargetFreqInt(gen_hz),
                })
                .map_err(|e| format!("init generator: {:?}", e))?;
            true
        } else {
            false
        };

        // Drop anything left over in the FIFO from a previous run.
        while self.fifo.csr.rf(bio_bdma::SFR_FLEVEL_PCLK_REGFIFO_LEVEL1) != 0 {
            let _ = self.fifo.csr.r(bio_bdma::SFR_RXF1);
        }

        self.bio.set_core_run_state(&self.grant, true);

        let mut raw = vec![0u32; n];
        let mut full_hits = 0u32;
        let mut err = None;
        // Skip the first few samples while both cores spin up.
        let mut skip = 16;
        let mut i = 0;
        while i < n {
            let mut spins = 0u32;
            let level = loop {
                let l = self.fifo.csr.rf(bio_bdma::SFR_FLEVEL_PCLK_REGFIFO_LEVEL1);
                if l != 0 {
                    break l;
                }
                spins += 1;
                if spins > STALL_SPINS {
                    break 0;
                }
            };
            if level == 0 {
                err = Some(format!("BIO stalled after {} samples", i));
                break;
            }
            // A full FIFO means the sampler may have halted and missed quanta.
            if level >= 8 {
                full_hits += 1;
            }
            let w = self.fifo.csr.r(bio_bdma::SFR_RXF1);
            if skip > 0 {
                skip -= 1;
                continue;
            }
            raw[i] = w;
            i += 1;
        }

        self.bio.set_core_run_state(&self.grant, false);
        self.bio.de_init_core(sampler).ok();
        self.bio.de_init_core(unused).ok();
        if gen_ok {
            self.bio.de_init_core(generator).ok();
        }
        match err {
            Some(e) => Err(e),
            None => Ok((raw.iter().map(|&w| pack(w)).collect(), actual, full_hits)),
        }
    }
}

/// BIO image = code, zero padding up to PAGE_TABLE_OFFSET (0x400), then `table` words.
fn with_table(code: &[u8], table: &[u32]) -> Result<Vec<u8>, String> {
    if code.len() > PAGE_TABLE_OFFSET {
        return Err("BIO code overlaps its data table".into());
    }
    let mut image = vec![0u8; PAGE_TABLE_OFFSET];
    image[..code.len()].copy_from_slice(code);
    for w in table {
        image.extend_from_slice(&w.to_le_bytes());
    }
    Ok(image)
}

fn pack(w: u32) -> u8 {
    let mut b = 0u8;
    for (i, &bit) in CH_BITS.iter().enumerate() {
        b |= (((w >> bit) & 1) as u8) << i;
    }
    b
}

fn send_all(usb: &UsbHid, mut data: &[u8]) {
    while !data.is_empty() {
        match usb.serial_send(data) {
            Ok(0) | Err(_) => xous::yield_slice(),
            Ok(sent) => data = &data[sent..],
        }
    }
}

const REC: usize = 512;
const REC_PAYLOAD: usize = REC - 8;
/// Records per serial_send call: 7 * 512 = 3584 bytes fits in one 3840-byte IPC page.
const RECS_PER_SEND: usize = 7;

struct Tx {
    seq: u16,
}

impl Tx {
    fn send_msg(&mut self, usb: &UsbHid, kind: &[u8; 4], body: &[u8]) { self.send_parts(usb, kind, &[body]); }

    /// Streams a message whose body is the concatenation of `parts`, building at most
    /// RECS_PER_SEND records (3.5 KiB) at a time - no full-size copies of the body.
    fn send_parts(&mut self, usb: &UsbHid, kind: &[u8; 4], parts: &[&[u8]]) {
        let len: usize = parts.iter().map(|p| p.len()).sum();
        let mut header = [0u8; 8];
        header[..4].copy_from_slice(kind);
        header[4..].copy_from_slice(&(len as u32).to_le_bytes());
        let mut bytes = header.iter().chain(parts.iter().flat_map(|p| p.iter())).copied().peekable();

        let mut out = Vec::with_capacity(REC * RECS_PER_SEND);
        while bytes.peek().is_some() {
            let start = out.len();
            out.extend_from_slice(&[0xB5, 0x5C]);
            out.extend_from_slice(&self.seq.to_le_bytes());
            out.extend(bytes.by_ref().take(REC_PAYLOAD));
            out.resize(start + REC - 4, 0);
            let crc = crc32fast::hash(&out[start..]);
            out.extend_from_slice(&crc.to_le_bytes());
            self.seq = self.seq.wrapping_add(1);
            if out.len() == REC * RECS_PER_SEND || bytes.peek().is_none() {
                send_all(usb, &out);
                out.clear();
            }
        }
    }
}

fn main() -> ! {
    log_server::init_wait().unwrap();
    log::set_max_level(log::LevelFilter::Info);
    let timer = ticktimer::Ticktimer::new().unwrap();
    let usb = UsbHid::new();
    log::info!("baoscope v{} PID={}", PROTOCOL_VERSION, xous::process::id());

    let mut scope = Scope::new().expect("could not claim BIO resources");

    while usb.status() != UsbDeviceState::Configured {
        timer.sleep_ms(10).ok();
    }
    log::info!("USB configured, waiting for commands");

    let mut tx = Tx { seq: 0 };
    // One STRM message fills exactly RECS_PER_SEND records.
    let filler: Vec<u8> = (0..RECS_PER_SEND * REC_PAYLOAD - 8).map(|i| i as u8).collect();

    loop {
        let line = usb.serial_wait_ascii(Some('\n'));
        let args: Vec<&str> = line.split_whitespace().collect();
        match args.as_slice() {
            ["ping"] => tx.send_msg(&usb, b"PONG", &[]),
            ["stream", batches] => {
                let batches: u32 = batches.parse().unwrap_or(1000);
                for _ in 0..batches {
                    tx.send_msg(&usb, b"STRM", &filler);
                }
            }
            // i2c <sample_hz> <bus_hz> <scl_bit> <nevents> <timeout_ms> <selftest 0|1>  (SDA = scl_bit+1)
            // spi <sample_hz> <sck_hz> <base_bit> <nevents> <timeout_ms> <selftest 0|1>  (SCK,MOSI,MISO,CS)
            [proto @ ("i2c" | "spi"), sample_hz, bus_hz, pin, n, timeout, selftest] => {
                let sample_hz: u32 = sample_hz.parse().unwrap_or(10_000_000);
                let bus_hz: u32 = bus_hz.parse().unwrap_or(100_000);
                let pin: u32 = pin.parse().unwrap_or(16);
                let n: usize = n.parse::<usize>().unwrap_or(1024).clamp(1, DMA_MAX_SAMPLES);
                let timeout: u64 = timeout.parse().unwrap_or(1000);
                let t0 = timer.elapsed_ms();
                // selftest: 0 = external signal, 1 = BIO test source, 2 = real I2C0 controller traffic
                let st = *selftest == "1";
                let stop = Arc::new(AtomicBool::new(false));
                let traffic = (*selftest == "2" && *proto == "i2c")
                    .then(|| start_traffic(RealTraffic::I2cScan, stop.clone()));
                let (kind, result) = if *proto == "spi" {
                    (b"SPI1", scope.decode_spi(sample_hz, bus_hz, pin, n, timeout, st, &timer))
                } else {
                    (b"I2C1", scope.decode_i2c(sample_hz, bus_hz, pin, n, timeout, st, &timer))
                };
                stop.store(true, Ordering::Relaxed);
                if let Some(h) = traffic {
                    log::info!("real I2C traffic: {} probes", h.join().unwrap_or(0));
                }
                match result {
                    Ok((words, actual, finished)) => {
                        let elapsed = (timer.elapsed_ms() - t0) as u32;
                        let mut hdr = [0u8; 20];
                        for (i, v) in [PROTOCOL_VERSION, actual, words.len() as u32, elapsed, finished as u32]
                            .iter()
                            .enumerate()
                        {
                            hdr[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                        }
                        let body: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                        tx.send_parts(&usb, kind, &[&hdr, &body]);
                    }
                    Err(e) => tx.send_msg(&usb, b"ERR_", e.as_bytes()),
                }
            }
            // uart <baud> <pin_bit> <nbytes> <timeout_ms> <selftest 0|1>
            ["uart", baud, pin, nbytes, timeout, selftest] => {
                let baud: u32 = baud.parse().unwrap_or(115_200);
                let pin: u32 = pin.parse().unwrap_or(16);
                let nbytes: usize = nbytes.parse::<usize>().unwrap_or(1024).clamp(1, DMA_MAX_SAMPLES);
                let timeout: u64 = timeout.parse().unwrap_or(1000);
                let t0 = timer.elapsed_ms();
                // selftest: 0 = external signal, 1 = BIO test source, 2 = real UART2 log traffic
                let stop = Arc::new(AtomicBool::new(false));
                let traffic = (*selftest == "2").then(|| start_traffic(RealTraffic::UartLog, stop.clone()));
                let result = scope.decode_uart(baud, pin, nbytes, timeout, *selftest == "1", &timer);
                stop.store(true, Ordering::Relaxed);
                if let Some(h) = traffic {
                    let _ = h.join();
                }
                match result {
                    Ok((words, actual, finished)) => {
                        let elapsed = (timer.elapsed_ms() - t0) as u32;
                        let mut hdr = [0u8; 20];
                        for (i, v) in [PROTOCOL_VERSION, actual / 8, words.len() as u32, elapsed, finished as u32]
                            .iter()
                            .enumerate()
                        {
                            hdr[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                        }
                        // 2 bytes per decoded byte: data, flags (bit 0 = framing error)
                        let body: Vec<u8> =
                            words.iter().flat_map(|&w| [w as u8, ((w >> 8) & 1) as u8]).collect();
                        tx.send_parts(&usb, b"UAR1", &[&hdr, &body]);
                    }
                    Err(e) => tx.send_msg(&usb, b"ERR_", e.as_bytes()),
                }
            }
            [cmd @ ("cap" | "dcap"), rate, n, rest @ ..] => {
                let dma = *cmd == "dcap";
                let rate: u32 = rate.parse().unwrap_or(1_000_000);
                // dcap <rate> <n> [gen_hz] [mode]: mode 1 = 32-bit words, 2 = 2x16-bit, 8 = 8x4-bit (ch0..3)
                let mode: usize = match rest.get(1).and_then(|m| m.parse().ok()) {
                    Some(m @ (1 | 2 | 8)) => m,
                    _ => 1,
                };
                // optional 5th arg: 1 = split sampler/writer cores, 0 = sampler stores directly
                let split = rest.get(2).map(|s| *s == "1").unwrap_or(false);
                let max = if dma { (DMA_MAX_SAMPLES * mode).min(MAX_OUT_SAMPLES) } else { MAX_SAMPLES };
                let n: usize = n.parse::<usize>().unwrap_or(4096).clamp(1, max);
                let gen_hz: u32 = rest.first().and_then(|g| g.parse().ok()).unwrap_or(rate / 8);
                let t0 = timer.elapsed_ms();
                let result = if dma {
                    scope.capture_dma(rate, n, gen_hz, mode, split, &timer)
                } else {
                    scope.capture(rate, n, gen_hz)
                };
                match result {
                    Ok((raw, actual, full_hits)) => {
                        let elapsed = (timer.elapsed_ms() - t0) as u32;
                        let mut hdr = [0u8; 20];
                        for (i, v) in
                            [PROTOCOL_VERSION, actual, raw.len() as u32, elapsed, full_hits].iter().enumerate()
                        {
                            hdr[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                        }
                        tx.send_parts(&usb, b"CAP1", &[&hdr, &raw]);
                        log::info!("cap {} @ {} Hz: {} ms, full_hits {}", raw.len(), actual, elapsed, full_hits);
                    }
                    Err(e) => {
                        log::error!("{}", e);
                        tx.send_msg(&usb, b"ERR_", e.as_bytes());
                    }
                }
            }
            [] => {}
            _ => tx.send_msg(&usb, b"ERR_", b"unknown command"),
        }
    }
}

// Sampler: every quantum, push the raw GPIO input word into FIFO1.
#[rustfmt::skip]
bio_code!(sampler_code, BSCP_SAMPLER_START, BSCP_SAMPLER_END,
  "20:",
    "mv x20, x0",   // halt until next quantum
    "mv x17, x21",  // GPIO inputs -> FIFO1 (core halts if FIFO is full)
    "j 20b"
);

// DMA sampler: walks the page table at IMEM 0x400 and stores one GPIO word per quantum
// straight into each physical page. When done, pushes elapsed BIO cycles into FIFO1.
// Errata notes: all I-type immediates below have imm[4:0] != 20 and no lui/auipc/jal
// has 16..19 in bits [19:15] (checked with libs/bio-lib/erratum_check.py).
#[rustfmt::skip]
bio_code!(dma_sampler_code, BSCP_DMA_START, BSCP_DMA_END,
    "mv   t2, x31",          // start cycle count ([29:0])
    "li   a3, 0x400",        // page table pointer
    "lw   a4, 0(a3)",        // page count
    "addi a3, a3, 4",
  "30:",
    "lw   a0, 0(a3)",        // next physical page
    "addi a3, a3, 4",
    "li   a1, 1024",         // words per page
  "20:",
    "mv   x20, x0",          // halt until next quantum
    "sw   x21, 0(a0)",       // GPIO inputs -> RAM
    "addi a0, a0, 4",
    "addi a1, a1, -1",
    "bnez a1, 20b",
    "addi a4, a4, -1",
    "bnez a4, 30b",
    "mv   t1, x31",
    "sub  t1, t1, t2",       // elapsed cycles
    "mv   x17, t1",          // done -> FIFO1
  "40:",
    "j    40b"
);

// REGISTER RULE for all BIO code here: only x1..x15 (ra, sp, gp, tp, t0-t2, s0, s1, a0-a5)
// are plain registers. x16-x31 are hardware (FIFOs, quantum, GPIO, events, core id), so
// t3-t6, a6, a7 and s2-s11 must never be used as scratch. Reading write-only x28/x29 returns 0.
//
// DMA sampler, 2 samples/word: word = (GPIO_a >> 16) | (GPIO_b & 0xFFFF0000).
// Same page-table walk as dma_sampler_code; half the bus writes.
#[rustfmt::skip]
bio_code!(dma_sampler2_code, BSCP_DMA2_START, BSCP_DMA2_END,
    "mv   t2, x31",
    "lui  s0, 0xffff0",      // s0 (x8) = 0xFFFF_0000
    "li   a3, 0x400",
    "lw   a4, 0(a3)",
    "addi a3, a3, 4",
  "30:",
    "lw   a0, 0(a3)",
    "addi a3, a3, 4",
    "li   a1, 1024",
  "20:",
    "mv   x20, x0",          // sample a
    "srli t0, x21, 16",
    "mv   x20, x0",          // sample b
    "and  t1, x21, s0",
    "or   t0, t0, t1",
    "sw   t0, 0(a0)",
    "addi a0, a0, 4",
    "addi a1, a1, -1",
    "bnez a1, 20b",
    "addi a4, a4, -1",
    "bnez a4, 30b",
    "mv   t1, x31",
    "sub  t1, t1, t2",
    "mv   x17, t1",
  "40:",
    "j    40b"
);

// DMA sampler, 8 samples/word of GPIO[19:16] (PC0..PC3). Shift register: each new nibble
// enters at bits [31:28] and older ones move down, so after 8 samples the earliest is in
// bits [3:0]. 5 instructions per sample, one bus write per 8 samples.
#[rustfmt::skip]
bio_code!(dma_sampler8_code, BSCP_DMA8_START, BSCP_DMA8_END,
    "mv   t2, x31",
    "lui  s1, 0xf0",         // s1 (x9) = 0x000F_0000
    "li   a3, 0x400",
    "lw   a4, 0(a3)",
    "addi a3, a3, 4",
  "30:",
    "lw   a0, 0(a3)",
    "addi a3, a3, 4",
    "li   a1, 1024",
  "20:",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "sw   t0, 0(a0)",
    "addi a0, a0, 4",
    "addi a1, a1, -1",
    "bnez a1, 20b",
    "addi a4, a4, -1",
    "bnez a4, 30b",
    "mv   t1, x31",
    "sub  t1, t1, t2",
    "mv   x17, t1",
  "40:",
    "j    40b"
);

// Pins-to-inputs: releases every BaoScope channel pin (BIO bits 16-19, 23, 24, 26, 27 =
// 0x0D8F_0000) to input, then idles. Used whenever an external device drives the pins.
#[rustfmt::skip]
bio_code!(inputs_code, BSCP_INP_START, BSCP_INP_END,
    "lui  a0, 0x0d8f0",
    "mv   x26, a0",
    "mv   x25, a0",                     // (x26 & x25) -> 1 makes those pins inputs
  "20:",
    "j    20b"
);

// Idle: parks a granted core that this capture doesn't use.
#[rustfmt::skip]
bio_code!(idle_code, BSCP_IDLE_START, BSCP_IDLE_END,
  "20:",
    "j 20b"
);

// ---- Split pipeline: sampler cores pack and push into FIFO2 (x18); the writer stores. ----

// Writer: walks the page table at IMEM 0x400 like dma_sampler_code, but each word comes from
// FIFO2 (the `sw` reads x18, blocking until the sampler has pushed one). Free-running clock.
#[rustfmt::skip]
bio_code!(writer_code, BSCP_WR_START, BSCP_WR_END,
    "mv   t2, x31",
    "li   a3, 0x400",
    "lw   a4, 0(a3)",
    "addi a3, a3, 4",
  "30:",
    "lw   a0, 0(a3)",
    "addi a3, a3, 4",
    "li   a1, 1024",
  "20:",
    "sw   x18, 0(a0)",       // FIFO2 -> RAM
    "addi a0, a0, 4",
    "addi a1, a1, -1",
    "bnez a1, 20b",
    "addi a4, a4, -1",
    "bnez a4, 30b",
    "mv   t1, x31",
    "sub  t1, t1, t2",
    "mv   x17, t1",          // done -> FIFO1
  "40:",
    "j    40b"
);

// Split sampler, 1 sample/word, unrolled x4 so the loop jump is amortised.
#[rustfmt::skip]
bio_code!(split_sampler1_code, BSCP_SS1_START, BSCP_SS1_END,
  "20:",
    "mv x20, x0", "mv x18, x21",
    "mv x20, x0", "mv x18, x21",
    "mv x20, x0", "mv x18, x21",
    "mv x20, x0", "mv x18, x21",
    "j 20b"
);

// Split sampler, 2 samples/word: (GPIO_a >> 16) | (GPIO_b & 0xFFFF0000) -> FIFO2.
#[rustfmt::skip]
bio_code!(split_sampler2_code, BSCP_SS2_START, BSCP_SS2_END,
    "lui  s0, 0xffff0",      // s0 (x8) = 0xFFFF_0000
  "20:",
    "mv x20, x0", "srli t0, x21, 16",
    "mv x20, x0", "and t1, x21, s0", "or x18, t0, t1",
    "mv x20, x0", "srli t0, x21, 16",
    "mv x20, x0", "and t1, x21, s0", "or x18, t0, t1",
    "j 20b"
);

// Split sampler, 8 x 4-bit samples/word (ch0..3), same shift-register packing as mode 8.
#[rustfmt::skip]
bio_code!(split_sampler8_code, BSCP_SS8_START, BSCP_SS8_END,
    "lui  s1, 0xf0",         // s1 (x9) = 0x000F_0000
  "20:",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or t0, t0, t1",
    // last sample pushes the finished word straight into FIFO2. t0 needn't keep it: the next
    // 8 samples shift t0 right by 32 bits in total, so its old contents fall out completely.
    "mv x20, x0", "and t1, x21, s1", "slli t1, t1, 12", "srli t0, t0, 4", "or x18, t0, t1",
    "j 20b"
);

// ---- On-chip UART decode ----

// UART RX, 8N1 LSB-first, quantum = 8 x baud. IMEM 0x400 holds the pin's BIO bit number.
// Idle: poll each quantum for low. Start: re-check at ~mid start bit (4 quanta after the edge
// quantum) to reject glitches. Data: sample every 8 quanta. Stop: sample, flag framing error if
// low, and wait for the line to return high before re-arming. Each byte -> FIFO2 (writer core).
#[rustfmt::skip]
bio_code!(uart_rx_code, BSCP_URX_START, BSCP_URX_END,
    "li   a3, 0x400",
    "lw   a5, 0(a3)",                   // a5 = pin bit
  "20:",
    "mv   x20, x0",
    "srl  t0, x21, a5",
    "andi t0, t0, 1",
    "bnez t0, 20b",                     // wait for start edge
    "mv x20, x0", "mv x20, x0", "mv x20, x0",
    "srl  t0, x21, a5",
    "andi t0, t0, 1",
    "bnez t0, 20b",                     // high again at mid start bit: glitch, re-arm
    "li   a1, 8",
    "li   a2, 0",
  "21:",
    "mv x20, x0", "mv x20, x0", "mv x20, x0", "mv x20, x0",
    "mv x20, x0", "mv x20, x0", "mv x20, x0", "mv x20, x0",
    "srl  t0, x21, a5",
    "andi t0, t0, 1",
    "srli a2, a2, 1",
    "slli t0, t0, 7",
    "or   a2, a2, t0",                  // LSB first: new bit enters at bit 7
    "addi a1, a1, -1",
    "bnez a1, 21b",
    "mv x20, x0", "mv x20, x0", "mv x20, x0", "mv x20, x0",
    "mv x20, x0", "mv x20, x0", "mv x20, x0", "mv x20, x0",
    "srl  t0, x21, a5",
    "andi t0, t0, 1",
    "xori t0, t0, 1",                   // 1 = stop bit was low = framing error
    "slli t1, t0, 8",
    "or   x18, a2, t1",                 // byte | ferr<<8 -> FIFO2
  "22:",
    "beqz t0, 20b",                     // good stop bit: straight back to idle
    "srl  t0, x21, a5",                 // framing error / break: wait for line high
    "andi t0, t0, 1",
    "xori t0, t0, 1",
    "j    22b"
);

// UART TX self-test source on the same pin, quantum = baud: 16 idle bits, then bytes
// 0,1,2,...,255,0,... each as start + 8 data (LSB first) + stop + 1 idle bit.
#[rustfmt::skip]
bio_code!(uart_tx_code, BSCP_UTX_START, BSCP_UTX_END,
    "li   a3, 0x400",
    "lw   a5, 0(a3)",                   // pin bit
    "li   a0, 1",
    "sll  a0, a0, a5",                  // a0 = pin mask
    "mv   x26, a0",
    "mv   x24, a0",                     // pin -> output
    "mv   x21, a0",                     // idle high
    "li   a4, 16",
  "19:",
    "mv   x20, x0",
    "addi a4, a4, -1",
    "bnez a4, 19b",
    "li   a1, 0",
  "20:",
    "mv   x21, x0",                     // start bit
    "mv   x20, x0",
    "mv   a2, a1",
    "li   a4, 8",
  "21:",
    "andi t0, a2, 1",
    "sll  t0, t0, a5",
    "mv   x21, t0",                     // data bit
    "mv   x20, x0",
    "srli a2, a2, 1",
    "addi a4, a4, -1",
    "bnez a4, 21b",
    "mv   x21, a0",                     // stop bit
    "mv   x20, x0",
    "mv   x20, x0",                     // 1 idle bit
    "addi a1, a1, 1",
    "andi a1, a1, 255",
    "j    20b"
);

// ---- On-chip I2C decode ----

// I2C RX: polls the 2-bit line state s = SCL | SDA<<1 (SCL = BIO bit at IMEM 0x400, SDA = next
// bit) every quantum and acts only on changes:
//   SCL rising            -> shift SDA into the byte; after 9 bits emit BYTE (data, ack)
//   SCL high, SDA falling -> START (also repeated START); resets the bit counter
//   SCL high, SDA rising  -> STOP
// Events -> FIFO2: tag<<16 | ack<<8 | data with tag 1 = START, 2 = STOP, 3 = BYTE.
// Registers: a5 SCL bit, a2 previous state, a1 bit count, a0 shift register,
// s0/s1/tp = START/STOP/BYTE tags, t0..t2 scratch (all within x1..x15).
#[rustfmt::skip]
bio_code!(i2c_rx_code, BSCP_IRX_START, BSCP_IRX_END,
    "li   a3, 0x400",
    "lw   a5, 0(a3)",
    "lui  s0, 0x10",                    // 1 << 16 START
    "lui  s1, 0x20",                    // 2 << 16 STOP
    "lui  tp, 0x30",                    // 3 << 16 BYTE
    "li   a2, 3",                       // bus idle: SCL and SDA high
    "li   a1, 0",
    "li   a0, 0",
  "20:",
    "mv   x20, x0",
    "srl  t0, x21, a5",
    "andi t0, t0, 3",
    "beq  t0, a2, 20b",                 // no change
    "xor  t1, t0, a2",                  // which lines changed
    "andi t2, t0, 1",
    "beqz t2, 29f",                     // SCL low now: nothing to decode, just track state
    "andi t1, t1, 1",
    "bnez t1, 25f",                     // SCL just rose: data bit
    "andi t2, t0, 2",                   // SCL steady high, SDA changed
    "bnez t2, 23f",
    "mv   x18, s0",                     // SDA fell: START
    "li   a1, 0",
    "li   a0, 0",
    "j    29f",
  "23:",
    "mv   x18, s1",                     // SDA rose: STOP
    "li   a1, 0",
    "li   a0, 0",
    "j    29f",
  "25:",
    "srli t2, t0, 1",                   // SDA value
    "slli a0, a0, 1",
    "or   a0, a0, t2",
    "addi a1, a1, 1",
    "li   t1, 9",
    "bne  a1, t1, 29f",
    "srli t2, a0, 1",                   // data = first 8 bits (MSB first)
    "andi t1, a0, 1",                   // ack = 9th bit
    "slli t1, t1, 8",
    "or   t2, t2, t1",
    "or   x18, t2, tp",                 // BYTE event
    "li   a1, 0",
    "li   a0, 0",
  "29:",
    "mv   a2, t0",
    "j    20b"
);

// I2C master test source (push-pull, same pins), quantum = 4 x bus rate, 4 quanta per bit.
// Repeats: START, 0xA0 +ACK, k +ACK, ~k +NACK, STOP, idle; k = 0,1,2,...
// Registers: a5 SCL bit, s0 SCL mask, s1 SDA mask, a3 k, a1 byte, a2 ack bit, a0 bit,
// t2 bit counter, gp saved ra (nested call), t0/t1 scratch.
#[rustfmt::skip]
bio_code!(i2c_tx_code, BSCP_ITX_START, BSCP_ITX_END,
    "li   a3, 0x400",
    "lw   a5, 0(a3)",
    "li   s0, 1",
    "sll  s0, s0, a5",                  // SCL mask
    "slli s1, s0, 1",                   // SDA mask
    "or   t0, s0, s1",
    "mv   x26, t0",
    "mv   x24, t0",                     // both pins -> outputs
    "mv   x21, t0",                     // idle: both high
    "li   t2, 16",
  "18:",
    "mv   x20, x0",
    "addi t2, t2, -1",
    "bnez t2, 18b",
    "li   a3, 0",
  "20:",
    "or   t0, s0, s1",                  // START: SDA falls while SCL high
    "mv   x21, t0",
    "mv   x20, x0",
    "mv   x20, x0",
    "mv   x21, s0",
    "mv   x20, x0",
    "mv   x20, x0",
    "mv   x21, x0",                     // SCL low
    "mv   x20, x0",
    "li   a1, 0xA0",                    // address 0x50, write
    "li   a2, 0",
    "jal  ra, 61f",
    "mv   a1, a3",                      // data k
    "li   a2, 0",
    "jal  ra, 61f",
    "xori a1, a3, 255",                 // data ~k, NACK
    "li   a2, 1",
    "jal  ra, 61f",
    "mv   x21, x0",                     // STOP: SDA rises while SCL high
    "mv   x20, x0",
    "mv   x21, s0",
    "mv   x20, x0",
    "mv   x20, x0",
    "or   t0, s0, s1",
    "mv   x21, t0",
    "mv   x20, x0", "mv x20, x0", "mv x20, x0", "mv x20, x0",
    "mv   x20, x0", "mv x20, x0", "mv x20, x0", "mv x20, x0",
    "addi a3, a3, 1",
    "andi a3, a3, 255",
    "j    20b",
  // send_bit(a0): SDA = a0 with SCL low, then SCL high for 2 quanta, then SCL low
  "60:",
    "sll  t0, a0, a5",
    "slli t0, t0, 1",                   // a0 at the SDA bit
    "mv   x21, t0",
    "mv   x20, x0",
    "mv   x20, x0",
    "or   t1, t0, s0",
    "mv   x21, t1",                     // SCL high: receiver samples here
    "mv   x20, x0",
    "mv   x20, x0",
    "mv   x21, t0",                     // SCL low, SDA held
    "ret",
  // send_byte(a1 = byte MSB first, a2 = ack bit)
  "61:",
    "mv   gp, ra",
    "li   t2, 8",
  "62:",
    "srli a0, a1, 7",
    "andi a0, a0, 1",
    "jal  ra, 60b",
    "slli a1, a1, 1",
    "addi t2, t2, -1",
    "bnez t2, 62b",
    "mv   a0, a2",
    "jal  ra, 60b",
    "mv   ra, gp",
    "ret"
);

// ---- On-chip SPI decode (mode 0: CPOL=0, CPHA=0, MSB first) ----

// SPI RX: polls the 4-bit state s = SCK | MOSI<<1 | MISO<<2 | CS<<3 (base BIO bit at IMEM
// 0x400) every quantum and acts only on change:
//   CS falls           -> ASSERT event, reset byte
//   CS rises           -> RELEASE event (low bits = leftover bit count of a partial byte)
//   SCK rises, CS low  -> shift MOSI and MISO in; after 8 bits emit BYTE
// Events -> FIFO2: tag<<16 | miso<<8 | mosi, tag 1 = ASSERT, 2 = RELEASE, 3 = BYTE.
// Registers: a5 base bit, a2 prev state, a3 bit count, a0 MOSI byte, a1 MISO byte, a4 = 8,
// s0/s1/tp tags, t0..t2 scratch.
#[rustfmt::skip]
bio_code!(spi_rx_code, BSCP_SRX_START, BSCP_SRX_END,
    "li   a3, 0x400",
    "lw   a5, 0(a3)",
    "lui  s0, 0x10",                    // ASSERT
    "lui  s1, 0x20",                    // RELEASE
    "lui  tp, 0x30",                    // BYTE
    "li   a2, 8",                       // idle: CS high, SCK low
    "li   a3, 0",
    "li   a0, 0",
    "li   a1, 0",
    "li   a4, 8",
  "20:",
    "mv   x20, x0",
    "srl  t0, x21, a5",
    "andi t0, t0, 15",
    "beq  t0, a2, 20b",
    "xor  t1, t0, a2",
    "andi t2, t1, 8",
    "beqz t2, 24f",                     // CS unchanged
    "andi t2, t0, 8",
    "bnez t2, 22f",
    "mv   x18, s0",                     // CS asserted
    "j    23f",
  "22:",
    "or   x18, s1, a3",                 // CS released (+ leftover bits)
  "23:",
    "li   a3, 0",
    "li   a0, 0",
    "li   a1, 0",
    "j    29f",
  "24:",
    "andi t2, t0, 8",
    "bnez t2, 29f",                     // CS high: not our transaction
    "andi t2, t1, 1",
    "beqz t2, 29f",                     // SCK unchanged
    "andi t2, t0, 1",
    "beqz t2, 29f",                     // SCK fell
    "srli t2, t0, 1",                   // SCK rose: sample MOSI
    "andi t2, t2, 1",
    "slli a0, a0, 1",
    "or   a0, a0, t2",
    "srli t2, t0, 2",                   // and MISO
    "andi t2, t2, 1",
    "slli a1, a1, 1",
    "or   a1, a1, t2",
    "addi a3, a3, 1",
    "bne  a3, a4, 29f",
    "slli t2, a1, 8",
    "or   t2, t2, a0",
    "or   x18, t2, tp",                 // BYTE event
    "li   a3, 0",
    "li   a0, 0",
    "li   a1, 0",
  "29:",
    "mv   a2, t0",
    "j    20b"
);

// SPI master test source (also plays the slave's MISO), push-pull on the same 4 pins,
// quantum = 2 x SCK. Repeats: CS low, (MOSI/MISO) 9F/FF, k/~k, A5/k, CS high, idle.
// Registers: a5 base bit, s0 4-pin mask, s1 SCK mask, gp CS mask, a3 k, a1/a2 MOSI/MISO byte,
// t2 bit counter, t0/t1 scratch.
#[rustfmt::skip]
bio_code!(spi_tx_code, BSCP_STX_START, BSCP_STX_END,
    "li   a3, 0x400",
    "lw   a5, 0(a3)",
    "li   s0, 15",
    "sll  s0, s0, a5",
    "li   s1, 1",
    "sll  s1, s1, a5",                  // SCK
    "li   gp, 8",
    "sll  gp, gp, a5",                  // CS
    "mv   x26, s0",
    "mv   x24, s0",                     // all 4 pins -> outputs
    "mv   x21, gp",                     // idle: CS high, SCK low
    "li   t2, 16",
  "18:",
    "mv   x20, x0",
    "addi t2, t2, -1",
    "bnez t2, 18b",
    "li   a3, 0",
  "20:",
    "mv   x21, x0",                     // CS low
    "mv   x20, x0",
    "li   a1, 0x9F",
    "li   a2, 0xFF",
    "jal  ra, 61f",
    "mv   a1, a3",
    "xori a2, a3, 255",
    "jal  ra, 61f",
    "li   a1, 0xA5",
    "mv   a2, a3",
    "jal  ra, 61f",
    "mv   x20, x0",
    "mv   x21, gp",                     // CS high
    "mv   x20, x0",
    "mv   x20, x0", "mv x20, x0", "mv x20, x0",
    "addi a3, a3, 1",
    "andi a3, a3, 255",
    "j    20b",
  // send_byte(a1 = MOSI, a2 = MISO), MSB first, mode 0: data set while SCK low, SCK high = sample
  "61:",
    "li   t2, 8",
  "62:",
    "srli t0, a1, 7",
    "andi t0, t0, 1",
    "slli t0, t0, 1",                   // MOSI -> bit 1
    "srli t1, a2, 7",
    "andi t1, t1, 1",
    "slli t1, t1, 2",                   // MISO -> bit 2
    "or   t0, t0, t1",                  // CS low, SCK low
    "sll  t0, t0, a5",
    "mv   x21, t0",
    "mv   x20, x0",
    "or   t1, t0, s1",
    "mv   x21, t1",                     // SCK high
    "mv   x20, x0",
    "slli a1, a1, 1",
    "slli a2, a2, 1",
    "addi t2, t2, -1",
    "bnez t2, 62b",
    "mv   x21, t0",                     // SCK low, CS still low
    "ret"
);

// Generator: 4-bit binary counter on BIO bits 16..19 (PC0..PC3), one count per quantum.
#[rustfmt::skip]
bio_code!(generator_code, BSCP_GEN_START, BSCP_GEN_END,
    "lui a0, 0xf0",     // a0 = 0x000F_0000: PC0..PC3
    "mv x26, a0",       // GPIO action mask
    "mv x24, a0",       // make masked pins outputs
    "li a1, 0",
  "20:",
    "slli a2, a1, 16",
    "mv x21, a2",       // (x26 & a2) -> pins
    "mv x20, x0",       // halt until next quantum
    "addi a1, a1, 1",
    "j 20b"
);
