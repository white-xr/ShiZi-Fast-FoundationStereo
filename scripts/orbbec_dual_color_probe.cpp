#include <libobsensor/ObSensor.hpp>

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace {

OBFormat parseFormat(const std::string &name) {
    if(name == "BGR") {
        return OB_FORMAT_BGR;
    }
    if(name == "RGB") {
        return OB_FORMAT_RGB;
    }
    if(name == "YUYV") {
        return OB_FORMAT_YUYV;
    }
    if(name == "MJPG") {
        return OB_FORMAT_MJPG;
    }
    throw std::runtime_error("Unsupported format: " + name);
}

std::string formatName(OBFormat format) {
    switch(format) {
    case OB_FORMAT_BGR:
        return "BGR";
    case OB_FORMAT_RGB:
        return "RGB";
    case OB_FORMAT_YUYV:
        return "YUYV";
    case OB_FORMAT_MJPG:
        return "MJPG";
    default:
        return std::to_string(static_cast<int>(format));
    }
}

cv::Mat frameToBgr(const std::shared_ptr<ob::Frame> &frame) {
    auto video = frame->as<ob::VideoFrame>();
    const int width = static_cast<int>(video->getWidth());
    const int height = static_cast<int>(video->getHeight());
    const auto format = video->getFormat();
    auto *data = video->getData();

    if(format == OB_FORMAT_BGR) {
        return cv::Mat(height, width, CV_8UC3, data).clone();
    }
    if(format == OB_FORMAT_RGB) {
        cv::Mat rgb(height, width, CV_8UC3, data);
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }
    if(format == OB_FORMAT_YUYV) {
        cv::Mat yuyv(height, width, CV_8UC2, data);
        cv::Mat bgr;
        cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
        return bgr;
    }
    if(format == OB_FORMAT_MJPG) {
        cv::Mat raw(1, static_cast<int>(video->getDataSize()), CV_8UC1, data);
        auto bgr = cv::imdecode(raw, cv::IMREAD_COLOR);
        if(bgr.empty()) {
            throw std::runtime_error("MJPG decode failed");
        }
        return bgr;
    }
    throw std::runtime_error("Unsupported runtime frame format: " + formatName(format));
}

void printStats(const std::string &label, const cv::Mat &image) {
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(image, mean, stddev);
    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(image.reshape(1), &minVal, &maxVal);
    std::cout << label << " shape=" << image.cols << "x" << image.rows << " channels=" << image.channels()
              << " min=" << minVal << " max=" << maxVal << " mean=(" << mean[0] << "," << mean[1] << "," << mean[2]
              << ") std=(" << stddev[0] << "," << stddev[1] << "," << stddev[2] << ")" << std::endl;
}

}  // namespace

int main(int argc, char **argv) try {
    const std::string outDir = argc > 1 ? argv[1] : "workspace/output/orbbec_dual_probe";
    const std::string formatText = argc > 2 ? argv[2] : "BGR";
    const int width = argc > 3 ? std::stoi(argv[3]) : 1280;
    const int height = argc > 4 ? std::stoi(argv[4]) : 800;
    const int fps = argc > 5 ? std::stoi(argv[5]) : 30;
    const OBFormat format = parseFormat(formatText);

    std::filesystem::create_directories(outDir);

    ob::Pipeline pipe;
    auto device = pipe.getDevice();
    std::cout << "Current preset before: " << device->getCurrentPresetName() << std::endl;
    device->loadPreset("Dual Color Streams");
    std::cout << "Current preset after: " << device->getCurrentPresetName() << std::endl;

    auto config = std::make_shared<ob::Config>();
    config->enableVideoStream(OB_SENSOR_COLOR_LEFT, width, height, fps, format);
    config->enableVideoStream(OB_SENSOR_COLOR_RIGHT, width, height, fps, format);
    config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);

    pipe.start(config);
    std::shared_ptr<ob::FrameSet> frameset;
    for(int i = 0; i < 60; ++i) {
        frameset = pipe.waitForFrameset(1000);
        if(frameset && frameset->getFrame(OB_FRAME_COLOR_LEFT) && frameset->getFrame(OB_FRAME_COLOR_RIGHT)) {
            break;
        }
    }
    if(!frameset) {
        throw std::runtime_error("No frameset received");
    }

    auto leftFrame = frameset->getFrame(OB_FRAME_COLOR_LEFT);
    auto rightFrame = frameset->getFrame(OB_FRAME_COLOR_RIGHT);
    if(!leftFrame || !rightFrame) {
        throw std::runtime_error("Frameset did not contain both left and right color frames");
    }

    auto left = frameToBgr(leftFrame);
    auto right = frameToBgr(rightFrame);
    printStats("left", left);
    printStats("right", right);

    cv::imwrite(outDir + "/left.png", left);
    cv::imwrite(outDir + "/right.png", right);
    if(left.size() != right.size()) {
        cv::resize(right, right, left.size());
    }
    cv::Mat preview;
    cv::hconcat(left, right, preview);
    cv::imwrite(outDir + "/preview.jpg", preview);
    std::cout << "Saved: " << outDir << "/preview.jpg" << std::endl;

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
