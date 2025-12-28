// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "CoreMLProfileApp",
    platforms: [
        .macOS(.v12)
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "CoreMLProfileApp",
            dependencies: [],
            path: "."
        )
    ]
)
