#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
// unlinked opencv_calib3d331d.lib
// unlinked opencv_features2d331d.lib
int main(int argc, char** argv)
{
	/*if (argc != 2)
	{
		cout << "Usage 002_hello_opencv.exe [image_to_load_and_display]\n";
		return -1;
	}*/
	//cout << argv[1] << endl;
	Mat image;
	image = imread("1.bmp", IMREAD_COLOR); // load the image
	if (NULL == image.data) // check for file errors
	{
		/*
		- why image is not loading???????????
		I change the format to window bitmap *.bmp
		which opencv always support with no needs of additional
		libraries.
		- How make imgread to read traditional formatas?????
		Read here https://docs.opencv.org/3.3.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
		about libjpg, libpng, libtiff, libjasper for know about read other more
		traditional formats. (see the notes section of imread method)
		*/
		cout << "Could not open or find the file\n";
		return -1;
	}
	namedWindow("display window", CV_WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("display window", image); // Show our image inside it.
	//waitKey(0); // Wait for a keystroke in the window
	return 0;
}