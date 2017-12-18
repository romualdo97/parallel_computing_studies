#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "Usage 002_hello_opencv.exe [image_to_load_and_display]\n";
		return -1;
	}
	cout << argv[1] << endl;
	Mat image;
	image = imread(argv[1], IMREAD_COLOR); // load the image
	if (image.empty()) // check for file errors
	{
		cout << "Could not open or find the file\n";
		return -1;
	}
	namedWindow("display window", CV_WINDOW_AUTOSIZE); // Create a window for display.
	imshow("display window", image); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}