import os
import numpy as np
import pyopencl as cl

class MarkesteijnOpenCLDemosaicer:
    def __init__(self, ctx=None, queue=None, kernel_path="data/kernels/demosaic_markesteijn.cl"):
        """
        Initializes the OpenCL Demosaicer.
        
        Args:
            ctx (pyopencl.Context, optional): An existing OpenCL context. If None, one will be created.
            queue (pyopencl.CommandQueue, optional): An existing OpenCL command queue. If None, one will be created.
            kernel_path (str): Path to the demosaic_markesteijn.cl kernel source code.
        """
        # 1. Initialize OpenCL Context and Command Queue
        if ctx is None:
            self.ctx = cl.create_some_context()
        else:
            self.ctx = ctx
            
        if queue is None:
            self.queue = cl.CommandQueue(self.ctx)
        else:
            self.queue = queue
            
        # 2. Compile OpenCL Kernels
        if not os.path.exists(kernel_path):
            raise FileNotFoundError(f"OpenCL kernel source not found at: {kernel_path}")
            
        with open(kernel_path, "r") as f:
            kernel_code = f.read()
            
        include_dir = os.path.abspath(os.path.dirname(kernel_path))
        options = ["-I", include_dir]
        self.prg = cl.Program(self.ctx, kernel_code).build(options=options)

        # Cache kernel references to prevent RepeatedKernelRetrieval warnings and improve performance
        self.k_initial_copy = self.prg.markesteijn_initial_copy
        self.k_green_minmax = self.prg.markesteijn_green_minmax
        self.k_interpolate_green = self.prg.markesteijn_interpolate_green
        self.k_recalculate_green = self.prg.markesteijn_recalculate_green
        self.k_solitary_green = self.prg.markesteijn_solitary_green
        self.k_red_and_blue = self.prg.markesteijn_red_and_blue
        self.k_interpolate_twoxtwo = self.prg.markesteijn_interpolate_twoxtwo
        self.k_convert_yuv = self.prg.markesteijn_convert_yuv
        self.k_differentiate = self.prg.markesteijn_differentiate
        self.k_homo_threshold = self.prg.markesteijn_homo_threshold
        self.k_homo_set = self.prg.markesteijn_homo_set
        self.k_homo_sum = self.prg.markesteijn_homo_sum
        self.k_homo_max = self.prg.markesteijn_homo_max
        self.k_homo_max_corr = self.prg.markesteijn_homo_max_corr
        self.k_homo_quench = self.prg.markesteijn_homo_quench
        self.k_zero = self.prg.markesteijn_zero
        self.k_accu = self.prg.markesteijn_accu
        self.k_final = self.prg.markesteijn_final

    def demosaic(self, raw_image: np.ndarray, xtrans_pattern: np.ndarray, passes: int = 3, 
                 black_level: float = None, white_level: float = None, crop: bool = False) -> np.ndarray:
        """
        Demosaics a 2D RAW raw_image using the Markesteijn X-Trans algorithm.
        
        Args:
            raw_image (np.ndarray): 2D array representing the raw sensor data (float32 or uint16/uint8).
            xtrans_pattern (np.ndarray): 6x6 array representing the Fuji X-Trans pattern (values: 0=Red, 1=Green, 2=Blue).
            passes (int): Number of demosaicing passes (typically 1 or 3).
            black_level (float, optional): If provided, used to normalize the raw sensor values.
            white_level (float, optional): If provided, used to normalize the raw sensor values.
            crop (bool): If True, crops the padded black border (width dependent on passes).
            
        Returns:
            np.ndarray: 3D float32 RGB array representing the demosaiced image (values in range [0, 1]).
        """
        # Ensure raw_image is a 2D float32 array
        raw_float = raw_image.astype(np.float32)
        height, width = raw_float.shape
        
        # Apply normalization if black/white levels are provided
        if black_level is not None and white_level is not None:
            normalized = (raw_float - black_level) / (white_level - black_level)
            normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
        else:
            normalized = raw_float

        # Format xtrans pattern matching the expected 6x6 structure
        xtrans = xtrans_pattern.astype(np.uint8)
        if xtrans.shape != (6, 6):
            raise ValueError("xtrans_pattern must be a 6x6 numpy array.")

        def FCNxtrans(row, col):
            return xtrans[(row + 600) % 6, (col + 600) % 6]

        # Geometry tables matching the Markesteijn algorithm
        orth = [1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1]
        patt = [
            [0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0],
            [0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1]
        ]

        allhex = np.zeros((3, 3, 8, 2), dtype=np.int8)
        sgreen = np.zeros(2, dtype=np.int8)

        # Precompute the hexagonal structure lookup tables
        for row in range(3):
            for col in range(3):
                ng = 0
                for d in range(0, 10, 2):
                    g = 1 if FCNxtrans(row, col) == 1 else 0
                    if FCNxtrans(row + orth[d] + 6, col + orth[d + 2] + 6) == 1:
                        ng = 0
                    else:
                        ng += 1
                    if ng == 4:
                        sgreen[0] = col
                        sgreen[1] = row
                    if ng == g + 1:
                        for c in range(8):
                            v = orth[d] * patt[g][c * 2] + orth[d + 1] * patt[g][c * 2 + 1]
                            h = orth[d + 2] * patt[g][c * 2] + orth[d + 3] * patt[g][c * 2 + 1]
                            allhex[row, col, c ^ (g * 2 & d), 0] = h
                            allhex[row, col, c ^ (g * 2 & d), 1] = v

        # Directional and padding constants
        ndir = 8 if passes > 1 else 4
        pad_tile = 17 if passes > 1 else 12

        PAD_G1_G3, PAD_G_INTERP, PAD_G_RECALC = 3, 3, 6
        pad_rb_g, pad_rb_br, pad_g22, pad_yuv, pad_homo = 5, 5, 4, 13, 15

        # 3. Allocate OpenCL Buffers & Device Memories
        dev_in = cl.Image(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                          cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), 
                          shape=(width, height), hostbuf=normalized)

        dev_rgbv = [cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=4 * width * height * 4) for _ in range(8)]
        dev_drv = [cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=width * height * 4) for _ in range(8)]
        dev_homo = [cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=width * height * 1) for _ in range(8)]
        dev_homosum = [cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=width * height * 1) for _ in range(8)]

        dev_gminmax = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=2 * width * height * 4)
        dev_aux = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=4 * width * height * 4)

        dev_xtrans = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=xtrans)
        dev_allhex = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=allhex)

        dev_out = cl.Image(self.ctx, cl.mem_flags.READ_WRITE, 
                           cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT), 
                           shape=(width, height))
        dev_tmptmp = cl.Image(self.ctx, cl.mem_flags.READ_WRITE, 
                              cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT), 
                              shape=(width, height))

        cl_char2 = np.dtype([('x', np.int8), ('y', np.int8)])
        sgreen_arg = np.array((sgreen[0], sgreen[1]), dtype=cl_char2)

        # Work group sizes (Pad global size to multiples of 16 to avoid INVALID_WORK_GROUP_SIZE)
        local_size = (16, 16)
        padded_width = ((width + 15) // 16) * 16
        padded_height = ((height + 15) // 16) * 16
        global_size = (padded_width, padded_height)

        # 4. Dispatch the Pipeline Kernels
        
        # Step A: Initial Copy
        self.k_initial_copy(self.queue, global_size, None,
                      dev_in, dev_rgbv[0], np.int32(width), np.int32(height), dev_xtrans)

        # Step B: Duplicate Initial RGBV buffers
        for c in range(1, 4):
            cl.enqueue_copy(self.queue, dev_rgbv[c], dev_rgbv[0], byte_count=4 * width * height * 4)

        # Step C: Green Min/Max
        local_mem_gminmax = cl.LocalMemory(4 * 22 * 22)
        self.k_green_minmax(self.queue, global_size, local_size,
                      dev_rgbv[0], dev_gminmax, np.int32(width), np.int32(height), np.int32(PAD_G1_G3),
                      sgreen_arg, dev_xtrans, dev_allhex, local_mem_gminmax)

        # Step D: Interpolate Green
        local_mem_interp = cl.LocalMemory(4 * 4 * 28 * 28)
        self.k_interpolate_green(self.queue, global_size, local_size,
                            dev_rgbv[0], dev_rgbv[1], dev_rgbv[2], dev_rgbv[3],
                            dev_gminmax, np.int32(width), np.int32(height),
                            np.int32(PAD_G_INTERP), sgreen_arg, dev_xtrans, dev_allhex,
                            local_mem_interp)

        # Step E: Multi-Pass Recalculations
        for pass_idx in range(passes):
            rgb_offset = 4 if pass_idx >= 1 else 0
            if pass_idx == 1:
                for c in range(4):
                    cl.enqueue_copy(self.queue, dev_rgbv[c + 4], dev_rgbv[c], byte_count=4 * width * height * 4)
                    
            if pass_idx > 0:
                self.k_recalculate_green(self.queue, global_size, None,
                                    dev_rgbv[rgb_offset + 0], dev_rgbv[rgb_offset + 1],
                                    dev_rgbv[rgb_offset + 2], dev_rgbv[rgb_offset + 3],
                                    dev_gminmax, np.int32(width), np.int32(height),
                                    np.int32(PAD_G_RECALC), sgreen_arg, dev_xtrans, dev_allhex)

            # Solitary Green
            local_mem_solitary = cl.LocalMemory(4 * 4 * 20 * 20)
            for d in range(6):
                i_val = 1 if (d % 2 == 0) else 0
                h_val = 0 if (d % 2 == 0) else 2
                dir_arg = np.array((i_val, i_val ^ 1), dtype=cl_char2)
                trgb_idx = rgb_offset + [0, 1, 2, 2, 3, 3][d]
                
                self.k_solitary_green(self.queue, global_size, local_size,
                                 dev_rgbv[trgb_idx], dev_aux, np.int32(width), np.int32(height), np.int32(pad_rb_g),
                                 np.int32(d), dir_arg, np.int32(h_val), sgreen_arg, dev_xtrans, local_mem_solitary)

            # Red and Blue Interpolation
            local_mem_rb = cl.LocalMemory(4 * 4 * 22 * 22)
            for d in range(4):
                self.k_red_and_blue(self.queue, global_size, local_size,
                               dev_rgbv[rgb_offset + d], np.int32(width), np.int32(height), np.int32(pad_rb_br),
                               np.int32(d), sgreen_arg, dev_xtrans, local_mem_rb)
                
            # Interpolate 2x2
            local_mem_g22 = cl.LocalMemory(4 * 4 * 20 * 20)
            for d in range(0, ndir, 2):
                n = d // 2
                self.k_interpolate_twoxtwo(self.queue, global_size, local_size,
                                      dev_rgbv[rgb_offset + n], np.int32(width), np.int32(height), np.int32(pad_g22),
                                      np.int32(d), sgreen_arg, dev_xtrans, dev_allhex, local_mem_g22)

        # Step F: Convert YUV and Differentiate
        local_mem_diff = cl.LocalMemory(4 * 4 * 18 * 18)
        for d in range(ndir):
            self.k_convert_yuv(self.queue, global_size, None,
                          dev_rgbv[d], dev_aux, np.int32(width), np.int32(height), np.int32(pad_yuv))
            self.k_differentiate(self.queue, global_size, local_size,
                            dev_aux, dev_drv[d], np.int32(width), np.int32(height), np.int32(pad_yuv),
                            np.int32(d), local_mem_diff)

        # Step G: Homogeneity Thresholding & Setup
        local_mem_homo = cl.LocalMemory(4 * 18 * 18)
        for d in range(ndir):
            self.k_homo_threshold(self.queue, global_size, None,
                             dev_drv[d], dev_aux, np.int32(width), np.int32(height), np.int32(pad_homo), np.int32(d))

        for d in range(ndir):
            self.k_homo_set(self.queue, global_size, local_size,
                       dev_drv[d], dev_aux, dev_homo[d], np.int32(width), np.int32(height), np.int32(pad_homo),
                       local_mem_homo)

        # Step H: Sum Homogeneity
        local_mem_homosum = cl.LocalMemory(1 * 20 * 20)
        for d in range(ndir):
            self.k_homo_sum(self.queue, global_size, local_size,
                       dev_homo[d], dev_homosum[d], np.int32(width), np.int32(height), np.int32(pad_tile),
                       local_mem_homosum)

        # Step I: Homogeneity Max Correlation
        for d in range(ndir):
            self.k_homo_max(self.queue, global_size, None,
                       dev_homosum[d], dev_aux, np.int32(width), np.int32(height), np.int32(pad_tile), np.int32(d))

        self.k_homo_max_corr(self.queue, global_size, None,
                        dev_aux, np.int32(width), np.int32(height), np.int32(pad_tile))

        # Step J: Homogeneity Quenching
        if passes > 1:
            for d in range(ndir - 4):
                self.k_homo_quench(self.queue, global_size, None,
                              dev_homosum[d], dev_homosum[d + 4], np.int32(width), np.int32(height), np.int32(pad_tile))

        # Step K: Accumulation Zero & Loops
        self.k_zero(self.queue, global_size, None, dev_out, np.int32(width), np.int32(height), np.int32(pad_tile))

        dev_t1, dev_t2 = dev_out, dev_tmptmp
        for d in range(ndir):
            self.k_accu(self.queue, global_size, None,
                   dev_t1, dev_t2, dev_rgbv[d], dev_homosum[d], dev_aux, np.int32(width), np.int32(height),
                   np.int32(pad_tile))
            dev_t1, dev_t2 = dev_t2, dev_t1

        if dev_t1 != dev_tmptmp:
            cl.enqueue_copy(self.queue, dev_tmptmp, dev_t1, dest_origin=(0,0), src_origin=(0,0), region=(width, height))

        # Step L: Final Smooth
        self.k_final(self.queue, global_size, None,
                dev_tmptmp, dev_out, np.int32(width), np.int32(height), np.int32(pad_tile))

        # 5. Read Back the Final Image from GPU
        output_host = np.zeros((height, width, 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, output_host, dev_out, origin=(0, 0), region=(width, height))

        # Pull out and return the demosaiced RGB channels of shape (H, W, 3)
        rgb_out = output_host[:, :, :3]
        if crop:
            return rgb_out[pad_tile:-pad_tile, pad_tile:-pad_tile, :]
        return rgb_out

if __name__ == "__main__":
    print("======================================================")
    print("   TESTING MARKESTEIJN OPENCL DEMOSAICER MODULE")
    print("======================================================")
    
    # 1. Instantiate demosaicer
    demosaic_cl_path = "data/kernels/demosaic_markesteijn.cl"
    demosaicer = MarkesteijnOpenCLDemosaicer(kernel_path=demosaic_cl_path)
    print("OpenCL Demosaicer compiled successfully!")
    
    # 2. Fuji X-Trans pattern definition
    xtrans_pattern = np.array([
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
        [0, 2, 1, 2, 0, 1]
    ], dtype=np.uint8)
    
    # 3. Handle raw input loading (or dummy array if not found)
    raw_path = "DSCF0049.RAF"
    if os.path.exists(raw_path):
        import rawpy
        from PIL import Image
        print(f"Found real RAW file: '{raw_path}'. Loading...")
        with rawpy.imread(raw_path) as raw:
            raw_image = raw.raw_image
            
        print("Running 3-pass demosaicing on real RAW input...")
        rgb_image = demosaicer.demosaic(
            raw_image=raw_image,
            xtrans_pattern=xtrans_pattern,
            passes=3,
            black_level=1023.0,
            white_level=16383.0
        )
        
        # Apply standard display gamma encoding (simple representation)
        clamped_out = np.clip(rgb_image, 0.0, 1.0)
        gamma_encoded = np.power(clamped_out, 1.0 / 2.2)
        scaled_out = (gamma_encoded * 255.0).astype(np.uint8)
        
        img = Image.fromarray(scaled_out)
        img.save("opencl_demosaic_output.png")
        print("Saved demosaiced real output to 'opencl_demosaic_output.png'.")
    else:
        # Create a synthetic image with non-multiple of 16 dimensions to verify the padding fix!
        # Let's use 2003x3005 dimensions to strictly challenge work group sizes!
        print("Real RAW file 'DSCF0049.RAF' not found.")
        print("Generating a synthetic 2003 x 3005 raw image to verify workgroup boundary checks...")
        np.random.seed(42)
        synthetic_raw = np.random.rand(2003, 3005).astype(np.float32)
        
        print("Running 3-pass demosaicing on synthetic input...")
        rgb_image = demosaicer.demosaic(
            raw_image=synthetic_raw,
            xtrans_pattern=xtrans_pattern,
            passes=3
        )
        print(f"Synthetic demosaicing succeeded! Output shape: {rgb_image.shape}")
        
    print("======================================================")
    print("                ALL TESTS PASSED!")
    print("======================================================")
