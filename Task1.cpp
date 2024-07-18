/* FACE DETECTION AND RECOGNITION

Develop an AI application that can detect and recognize faces in
images or videos. Use pre-trained face detection models like Haar
cascades or deep learning-based face detectors, and optionally
add face recognition capabilities using techniques like Siamese
networks or ArcFace.
*/

#include <opencv2/opencv.hpp> // Include OpenCV library for image processing.
#include <dlib/opencv.h> // Include Dlib functions for interoperability with OpenCV.
#include <dlib/image_processing.h> // Include Dlib's image processing functions.
#include <dlib/dnn.h> // Include Dlib's deep neural network functions.
#include <dlib/data_io.h> // Include Dlib's data I/O functions.

using namespace cv; // Use the OpenCV namespace.
using namespace dlib; // Use the Dlib namespace.

// Define a simple ResNet network.
template <template <typename> class TAG, typename SUBNET>
using block = add_tag_layer<TAG, con<32, 3, 3, 1, 1, relu<con<32, 3, 3, 1, 1, SUBNET>>>>; // Define a block in the network with convolutional layers.

template <typename SUBNET> using level1 = relu<block<tag1, SUBNET>>; // Define level1 in the network with a relu activation function.
template <typename SUBNET> using level2 = relu<block<tag2, level1<SUBNET>>>; // Define level2 in the network.
template <typename SUBNET> using level3 = relu<block<tag3, level2<SUBNET>>>; // Define level3 in the network.

using net_type = loss_metric<fc_no_bias<128, avg_pool_everything<level3<max_pool<2, 2, 2, 2, input_rgb_image_sized<150>>>>>>; // Define the network type.

int main() {
    // Load the face detection model.
    CascadeClassifier face_cascade; // Create a CascadeClassifier object for face detection.
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) { // Load the Haar cascade model for face detection.
        std::cerr << "Error loading Haar cascade" << std::endl; // Print error message if the model fails to load.
        return -1; // Exit the program with an error code.
    }

    // Load the shape predictor model.
    shape_predictor sp; // Create a shape_predictor object.
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp; // Load the shape predictor model from file.

    // Load the face recognition model.
    net_type net; // Create a network object.
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net; // Load the ResNet model for face recognition from file.

    // Open the default camera.
    VideoCapture cap(0); // Create a VideoCapture object to access the camera.
    if (!cap.isOpened()) { // Check if the camera is successfully opened.
        std::cerr << "Unable to connect to camera" << std::endl; // Print error message if the camera fails to open.
        return 1; // Exit the program with an error code.
    }

    Mat frame; // Declare a Mat object to store the video frame.
    while (true) { // Start an infinite loop to process the video frames.
        cap >> frame; // Capture a frame from the camera.
        std::vector<Rect> faces; // Declare a vector to store detected faces.
        Mat gray; // Declare a Mat object to store the grayscale image.
        cvtColor(frame, gray, COLOR_BGR2GRAY); // Convert the captured frame to grayscale.
        face_cascade.detectMultiScale(gray, faces); // Detect faces in the grayscale image.

        std::vector<matrix<rgb_pixel>> face_chips; // Declare a vector to store face chips.
        for (auto face : faces) { // Loop through each detected face.
            rectangle rect(Point(face.x, face.y), Point(face.x + face.width, face.y + face.height)); // Create a rectangle around the detected face.
            rectangle(frame, rect, Scalar(0, 255, 0), 2); // Draw the rectangle on the frame.

            // Convert OpenCV rectangle to Dlib rectangle.
            dlib::rectangle dlib_rect(
                (long)face.x, (long)face.y,
                (long)(face.x + face.width), (long)(face.y + face.height)
            );

            cv_image<bgr_pixel> cimg(frame); // Convert OpenCV Mat to Dlib image.
            full_object_detection shape = sp(cimg, dlib_rect); // Detect facial landmarks.
            matrix<rgb_pixel> face_chip; // Declare a matrix to store the face chip.
            extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip); // Extract the face chip.
            face_chips.push_back(std::move(face_chip)); // Add the face chip to the vector.
        }

        if (face_chips.size() != 0) { // Check if any face chips were detected.
            std::vector<matrix<float, 0, 1>> face_descriptors = net(face_chips); // Compute the face descriptors.
            // Process face descriptors for recognition.
            // ...
        }

        // Display the frame.
        imshow("Webcam", frame); // Show the frame in a window.
        if (waitKey(1) >= 0) break; // Break the loop if a key is pressed.
    }

    return 0; // Return success code.
}
