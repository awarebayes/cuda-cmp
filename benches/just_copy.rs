// Import necessary crates
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cudarc::driver::{CudaDevice, CudaSlice};

fn just_copy_roundtrip_benchmark(c: &mut Criterion) {
    // Input image sizes
    let sizes = [(1920, 1080), (640, 360)];

    // Initialize CUDA device
    let device = CudaDevice::new(0).expect("Failed to initialize CUDA device");

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
                    let _d_input: CudaSlice<u8> = black_box(
                        device
                            .htod_copy(input_images)
                            .expect("Failed to allocate device memory for input"),
                    );
                    let d_output: CudaSlice<u8> = black_box(
                        device
                            .alloc_zeros::<u8>(num_pixels * 3 * batch_size)
                            .expect("Failed to allocate device memory for output"),
                    );

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

criterion_group!(benches, just_copy_roundtrip_benchmark);
criterion_main!(benches);
