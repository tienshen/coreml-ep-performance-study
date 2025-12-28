# CoreMLProfileApp

A minimal macOS command-line Swift app to load and run a Core ML model (.mlpackage or .mlmodel) for profiling in Xcode Instruments.

## Usage
- Accepts the model path as a command-line argument
- Runs inference on random input
- Prints timing results
- Use Xcode Instruments (Core ML or Time Profiler) to profile

## How to build and run
1. Open the project in Xcode
2. Build the command-line tool
3. Run with the path to your Core ML model as an argument
4. Use Xcode Instruments to profile the run
