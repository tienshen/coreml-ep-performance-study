import Foundation
import CoreML

func printUsage() {
    print("Usage: CoreMLProfileApp <model_path> [batch_size]")
}

func randomInput(shape: [Int], dtype: String = "fp32") -> MLMultiArray? {
    let count = shape.reduce(1, *)
    let type: MLMultiArrayDataType = (dtype == "fp16") ? .float16 : .float32
    guard let arr = try? MLMultiArray(shape: shape as [NSNumber], dataType: type) else { return nil }
    for i in 0..<count {
        arr[i] = NSNumber(value: Float.random(in: 0...1))
    }
    return arr
}

let args = CommandLine.arguments
if args.count < 2 {
    printUsage()
    exit(1)
}
let modelPath = args[1]
let batchSize = (args.count > 2) ? Int(args[2]) ?? 1 : 1

let modelURL = URL(fileURLWithPath: modelPath)
let config = MLModelConfiguration()
config.computeUnits = .all

guard let model = try? MLModel(contentsOf: modelURL, configuration: config) else {
    print("Failed to load model at \(modelPath)")
    exit(1)
}

// Get input description
let inputDesc = model.modelDescription.inputDescriptionsByName
let inputName = inputDesc.keys.first!
let inputShape = inputDesc[inputName]?.multiArrayConstraint?.shape.map { $0.intValue } ?? [batchSize, 3, 224, 224]
let dtype = (inputDesc[inputName]?.multiArrayConstraint?.dataType == .float16) ? "fp16" : "fp32"

guard let inputArray = randomInput(shape: inputShape, dtype: dtype) else {
    print("Failed to create input array")
    exit(1)
}

// Warmup
for _ in 0..<5 {
    _ = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: inputArray]))
}

// Timed runs
let numRuns = 50
var latencies: [Double] = []
for _ in 0..<numRuns {
    let start = DispatchTime.now()
    _ = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: inputArray]))
    let end = DispatchTime.now()
    let elapsed = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
    latencies.append(elapsed)
}
let mean = latencies.reduce(0, +) / Double(numRuns)
print("\nModel: \(modelPath)")
print("Batch size: \(batchSize)")
print("Mean latency: \(String(format: "%.3f", mean)) ms over \(numRuns) runs")
