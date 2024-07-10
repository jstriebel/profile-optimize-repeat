#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;
using namespace py::literals;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

py::array_t<uint8_t> colorcode_probabilities(py::array_t<double> probabilities, py::array_t<uint8_t> colors) {
    auto shape = probabilities.shape();
    auto probabilities_ptr = static_cast<double*>(probabilities.request().ptr);
    auto colors_ptr = static_cast<uint8_t*>(colors.request().ptr);
    auto colored_image = py::array_t<uint8_t>({int(shape[0]), int(shape[1]), 3});
    auto colored_image_ptr = static_cast<uint8_t*>(colored_image.request().ptr);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            double max_probability = 0.0;
            size_t max_index = 0;
            for (int k = 0; k < shape[2]; k++) {
                auto probability = *probabilities_ptr++;
                if (probability > max_probability) {
                    max_probability = probability;
                    max_index = k;
                }
            }
            auto class_color_ptr = colors_ptr + (max_index * 3);
            *colored_image_ptr++ = static_cast<uint8_t>(*class_color_ptr++ * max_probability);
            *colored_image_ptr++ = static_cast<uint8_t>(*class_color_ptr++ * max_probability);
            *colored_image_ptr++ = static_cast<uint8_t>(*class_color_ptr++ * max_probability);
        }
    }
    return colored_image;
}


PYBIND11_MODULE(pybind11_analysis, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin for colorcoding probabilities
    )pbdoc";

    m.def("colorcode_probabilities", &colorcode_probabilities, "probabilities"_a, "colors"_a, R"pbdoc(
        Generate colored image from probabilities and colors.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}