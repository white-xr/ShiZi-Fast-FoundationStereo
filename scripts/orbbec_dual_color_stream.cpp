#include <libobsensor/ObSensor.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

namespace {

#pragma pack(push, 1)
struct PacketHeader {
    char magic[4];
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t leftBytes;
    uint32_t rightBytes;
    uint64_t frameIndex;
    uint64_t leftFrameIndex;
    uint64_t rightFrameIndex;
    uint64_t leftTimestampUs;
    uint64_t rightTimestampUs;
    uint64_t leftSystemTimestampUs;
    uint64_t rightSystemTimestampUs;
};
#pragma pack(pop)

std::shared_ptr<ob::VideoFrame> getVideoFrame(const std::shared_ptr<ob::FrameSet> &frameset, OBFrameType type) {
    auto frame = frameset->getFrame(type);
    if(!frame) {
        return nullptr;
    }
    return frame->as<ob::VideoFrame>();
}

}  // namespace

int main(int argc, char **argv) try {
    const int width = argc > 1 ? std::stoi(argv[1]) : 1280;
    const int height = argc > 2 ? std::stoi(argv[2]) : 800;
    const int fps = argc > 3 ? std::stoi(argv[3]) : 30;
    const std::string preset = argc > 4 ? argv[4] : "Dual Color Streams";

    ob::Pipeline pipe;
    auto device = pipe.getDevice();
    device->loadPreset(preset.c_str());
    std::cerr << "Orbbec preset: " << device->getCurrentPresetName() << std::endl;

    auto config = std::make_shared<ob::Config>();
    config->enableVideoStream(OB_SENSOR_COLOR_LEFT, width, height, fps, OB_FORMAT_BGR);
    config->enableVideoStream(OB_SENSOR_COLOR_RIGHT, width, height, fps, OB_FORMAT_BGR);
    config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    pipe.enableFrameSync();
    pipe.start(config);
    std::cerr << "Orbbec dual color stream started: " << width << "x" << height << "@" << fps << " BGR" << std::endl;

    uint64_t frameIndex = 0;
    while(std::cout.good()) {
        auto frameset = pipe.waitForFrameset(1000);
        if(!frameset) {
            continue;
        }
        auto left = getVideoFrame(frameset, OB_FRAME_COLOR_LEFT);
        auto right = getVideoFrame(frameset, OB_FRAME_COLOR_RIGHT);
        if(!left || !right) {
            continue;
        }
        if(left->getFormat() != OB_FORMAT_BGR || right->getFormat() != OB_FORMAT_BGR) {
            std::cerr << "Unexpected frame format; expected BGR" << std::endl;
            continue;
        }

        const uint32_t leftBytes = left->getDataSize();
        const uint32_t rightBytes = right->getDataSize();
        PacketHeader header{{'O', 'B', 'L', '2'},
                            static_cast<uint32_t>(left->getWidth()),
                            static_cast<uint32_t>(left->getHeight()),
                            3,
                            leftBytes,
                            rightBytes,
                            ++frameIndex,
                            left->getIndex(),
                            right->getIndex(),
                            left->getTimeStampUs(),
                            right->getTimeStampUs(),
                            left->getSystemTimeStampUs(),
                            right->getSystemTimeStampUs()};
        std::cout.write(reinterpret_cast<const char *>(&header), sizeof(header));
        std::cout.write(reinterpret_cast<const char *>(left->getData()), leftBytes);
        std::cout.write(reinterpret_cast<const char *>(right->getData()), rightBytes);
        std::cout.flush();
    }

    pipe.stop();
    return 0;
}
catch(ob::Error &e) {
    std::cerr << "Orbbec error\nfunction:" << e.getFunction() << "\nargs:" << e.getArgs() << "\nmessage:" << e.what()
              << "\nstatus:" << e.getStatus() << "\ntype:" << e.getExceptionType() << std::endl;
    return 2;
}
catch(std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
}
