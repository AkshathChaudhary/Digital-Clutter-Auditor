from pathlib import Path
from benchmark.make_benchmark import create_text_fixtures

output_dir = Path("data/benchmark/files")

ground_truth = create_text_fixtures(output_dir)

print("Generated files:")
for file_path in output_dir.iterdir():
    print(file_path.name)

print("\nGround truth:")
print(ground_truth)

assert (
    (output_dir / "exact_original.txt").read_text()
    ==
    (output_dir / "exact_copy.txt").read_text()
)

print("\nBenchmark text fixtures generated successfully!")