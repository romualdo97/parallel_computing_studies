#include <opencv2\opencv.hpp>

#define SHOW_WEBCAM

using namespace cv;
using namespace std;

bool showImage();
bool showWebCam();

int main(int argc, char** argv)
{

#ifdef SHOW_WEBCAM
	if (!showWebCam()) return -1;
#else
	if (!showImage()) return -1;
#endif

	return 0;
}

bool showImage()
{
	Mat image;
	image = imread("1.jpg", IMREAD_COLOR); // load the image
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
		return false;
	}
	namedWindow("display window", CV_WINDOW_AUTOSIZE); // Create a window for display.
	imshow("display window", image); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return true;
}

bool showWebCam()
{
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "Sorry, can't use webcam" << std::endl;
		return false;
	}
	while (true)
	{
		Mat frame;
		cap >> frame;
		imshow("Frame", frame);
		if (waitKey(30) == 'q')
		{
			break;
		}
	}
	return true;
}