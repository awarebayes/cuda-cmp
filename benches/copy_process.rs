// Import necessary crates
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cudarc::{
    driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

fn process_images_benchmark(c: &mut Criterion) {
    // Input image sizes
    let sizes = [(1920, 1080), (640, 360)];

    // Initialize CUDA device
    let device = CudaDevice::new(0).expect("Failed to initialize CUDA device");

    // Load the kernel from a pre-compiled PTX file
    device
        .load_ptx(
            Ptx::from_file("./process_image.ptx"),
            "process_image_module",
            &["process_image"],
        )
        .expect("Failed to load PTX file");

    // Retrieve the function from the module
    let kernel = device
        .get_func("process_image_module", "process_image")
        .expect("Failed to get kernel function");

    let mut group = c.benchmark_group("Image Processing Comparison");

    for &(width, height) in &sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &(width, height),
            |b, &(width, height)| {
                b.iter(|| {
                    let num_pixels = black_box(width * height);

                    // Simulate image data for a batch of 100 images
                    let batch_size = black_box(100);
                    let input_images: Vec<u8> = vec![255; num_pixels * 3 * batch_size]; // Batch of RGB images
                    let mut output_images: Vec<u8> = vec![0; num_pixels * 3 * batch_size];

                    // Allocate GPU memory for batch
                    let d_input: CudaSlice<u8> = device
                        .htod_copy(input_images)
                        .expect("Failed to allocate device memory for input");
                    let mut d_output: CudaSlice<u8> = device
                        .alloc_zeros::<u8>(num_pixels * 3 * batch_size)
                        .expect("Failed to allocate device memory for output");

                    // Launch kernel
                    let config = LaunchConfig::for_num_elems((num_pixels * batch_size) as u32);

                    unsafe {
                        kernel
                            .clone()
                            .launch(
                                config,
                                (
                                    &mut d_output,
                                    &d_input,
                                    width as i32,
                                    height as i32,
                                    batch_size as i32,
                                ),
                            )
                            .expect("Failed to launch kernel");
                    }

                    // Copy output images back to host
                    device
                        .dtoh_sync_copy_into(&d_output, &mut output_images)
                        .expect("Failed to copy output to host");

                    black_box(());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, process_images_benchmark);
criterion_main!(benches);
