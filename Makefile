.PHONY: test cycles trace clean correctness

# Run the cycle count test (main benchmark)
cycles:
	python perf_takehome.py Tests.test_kernel_cycles

# Run correctness tests against submission harness
correctness:
	python tests/submission_tests.py CorrectnessTests

# Run all submission tests (correctness + speed thresholds)
test:
	python tests/submission_tests.py

# Generate a trace for visualization in Perfetto
trace:
	python perf_takehome.py Tests.test_kernel_trace
	@echo "Run 'make watch' in another terminal, then open http://localhost:8000"

# Start the trace viewer server
watch:
	python watch_trace.py

# Run reference kernel tests
ref:
	python perf_takehome.py Tests.test_ref_kernels

# Clean generated files
clean:
	rm -f trace.json
	rm -f __pycache__/*.pyc
	rm -rf __pycache__
	rm -rf tests/__pycache__
