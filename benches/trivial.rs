use criterion::{criterion_group, criterion_main, Criterion};

fn trivial_benchmark(c: &mut Criterion) {
    c.bench_function("simple_addition", |b| {
        b.iter(|| {
            let _result = 1 + std::hint::black_box(1);
            println!("aboba");
        });
    });
}

criterion_group!(benches, trivial_benchmark);
criterion_main!(benches);
