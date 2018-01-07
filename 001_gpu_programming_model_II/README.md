# Building OpenCV and running Hello World Example in 2018

In this tutorial we’re going to learn how to build OpenCV 3.3 static libraries from source code and setup our visual studio 2017 solution for run the below OpenCV hello world code.

We will do this for the **problem set 1** of **Introduction to Parallel Programming course** at Udacity, but this material is also useful as reference or tutorial explaining the process of building OpenCV and running the HelloWorld application from scratch in windows systems.

	#include <opencv2/core.hpp>
	#include <opencv2/imgcodecs.hpp>
	#include <opencv2/highgui.hpp>
	#include <iostream>
	using namespace cv;
	using namespace std;
	int main( int argc, char** argv )
	{
	    if( argc != 2)
	    {
	     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
	     return -1;
	    }
	    Mat image;
	    image = imread(argv[1], IMREAD_COLOR); // Read the file
	    if( image.empty() ) // Check for invalid input
	    {
	        cout << "Could not open or find the image" << std::endl ;
	        return -1;
	    }
	    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
	    imshow( "Display window", image ); // Show our image inside it.
	    waitKey(0); // Wait for a keystroke in the window
	    return 0;
	}

### **STEP 0)** Download and install a C++ compiler installing the Visual C++ packages for visual studio 2017

### **STEP 1)** Download CMake

Don´t forget to add CMake to the PATH during installation (is not mandatory but I recommend it, maybe someday you could need to write console scripts that usea CMake),  I will use the binary distribution of CMake for Win32-x86. [download from here](https://cmake.org/download/)

![enter image description here](https://i.imgur.com/CjIxtdu.png)

### **STEP 2)** Create a dir for download OpenCV source package

![enter image description here](https://i.imgur.com/VrRAxQR.png)

### **STEP 3)** Download OpenCV source package from [git repository](https://github.com/opencv/opencv)

![enter image description here](https://i.imgur.com/trNGgEI.png)

### **STEP 4)** Download dependencies for build Highgui module

In the above hello world code we can see an include for **opencv2/highgui**, this is an **OpenCV module** (as opencv2/core), if we want to use this fancy **module for create windows, show images** (highhui is responsible of namedWindow, imgRead) in windows and others useful GUI features we need to [download](https://www1.qt.io/download-open-source/?hsCtaTracking=f977210e-de67-475f-a32b-65cec207fd03%7Cd62710cd-e1db-46aa-8d4d-2f1c1ffdacea#section-2) and build the QT library. Go QT main page then click in **Qt Offline Instalers > Source packages & Other releases > For Windows users** as a single **zip file (626 MB)**.

![enter image description here](https://i.imgur.com/ouzxHHO.png)

### **STEP 5)** Extract it into a nice and short named directory like `D:/OpenCV/dep/qt/`

![enter image description here](https://i.imgur.com/I8oXmzl.png)

### **STEP 6)** Start VisualStudio Command Prompt

![enter image description here](https://i.imgur.com/NuBBd95.png)

next go to your extracted directory. In my case 

	> d:
	> cd D:\opencv-master\dep\qt-everywhere-opensource-src-5.9.1

### **STEP 8)** configure QT build
QT has many modules, but for compile OpenCV Highgui module we just need to compile the `qtbase` module, so let's configure our build before compilation using the `Developer Command Prompt for VisualStudio`.

	> configure -skip qt3d -skip qtactiveqt -skip qtandroidextras -skip qtcanvas3d -skip qtcharts -skip qtconnectivity -skip qtdatavis3d -skip qtdeclarative -skip qtdoc -skip qtgamepad -skip qtgraphicaleffects -skip qtimageformats -skip qtlocation -skip qtmacextras -skip qtmultimedia -skip qtnetworkauth -skip qtpurchasing -skip qtquickcontrols -skip qtquickcontrols2 -skip qtremoteobjects -skip qtscript -skip qtscxml -skip qtsensors -skip qtserialbus -skip qtserialport -skip qtspeech -skip qtsvg -skip qttools -skip qttranslations -skip qtvirtualkeyboard -skip qtwayland -skip qtwebchannel -skip qtwebengine -skip qtwebsockets -skip qtwebview -skip qtwinextras -skip qtx11extras -skip qtxmlpatterns

### **STEP 9)** Select the OpenSource QT edition

![enter image description here](https://i.imgur.com/hJnwIlq.png)

Then accept the **licence offer** typing `y` and enter.

> **When configuration error occurs:** run `nmake confclean` and run `configure` again.

### **STEP 10)** Build QT

Enter the next command for compile `qtbase` module.

	> nmake

### **STEP 11)** Set QT enviroment variables

Set an enviroment variable called `QTDIR` pointing to your QT extracted folder.

	setx -m QTDIR D:\opencv-master\dep\qt-everywhere-opensource-src-5.9.1

Set another enviroment variable called `QT_QPA_PLATFORM_PLUGIN_PATH`pointing to your `qt\qtbase\plugins` folder

	setx -m QT_QPA_PLATFORM_PLUGIN_PATH D:\opencv-master\dep\qt-everywhere-opensource-src-5.9.1\qtbase\plugins

If you don´t want to use the visual studio prompt you could also **right click on my pc > advanced system configuration > enviroment variables**  and under system variables (not user variables) click new and set its respective name to `QTDIR` and value to your extracted QT folder.

### **STEP 12)** Set `qtbase/bin` folder into `path`

![enter image description here](https://i.imgur.com/nQWHmYJ.png)

### **STEP 13)** Configuring CMake for build VisualStudio solution

![enter image description here](https://i.imgur.com/2kB0GtF.png)

For some of the packages CMake may not find all of the required files or directories. In case of these, CMake will throw an error in its output window (located at the bottom of the GUI) and set its field values to `NOTFOUND` constants.

![enter image description here](https://i.imgur.com/tDyAzwb.png)

For now let´s ignore this and let´s focus our attention in selecting the Grouped checkbox under the binary directory selection.

![enter image description here](https://i.imgur.com/JnnAh8c.png)

### **STEP 14)** Configuring CMake for build VisualStudio solution, again!

Now you should see a list of grouped options like this below.

![enter image description here](https://i.imgur.com/kzGAsvr.png)

The following is the basic CMake configuration for generate the essential depencies needed in the hello world code.

Also note that following OpenCV configuration will be able to read just images with `.bmp` format, I think you can play with configuration from this point.

**Check only the following options:** What is not mentioned here should be unchecked (just modify the groups mentioned).

	WITH:
		WITH_QT
	
	ENABLE:
		ENABLE_PRECOMPILED_HEADERS
		ENABLE_SOLUTION_FOLDERS
	
	BUILD:	
		BUILD_opencv_core
		BUILD_ZLIB
		BUILD_opencv_highgui
		BUILD_opencv_imgcodecs
		BUILD_opencv_imgproc

Then go to your `build\vs_15_2017` folder and open `OpenCV.sln` for **compile the solution.** (sorry my vs2017 is in spanish)

![enter image description here](https://i.imgur.com/boWegJv.png)

### **STEP 15)** Grouping dependencies

Now that we have compiled all the needed opencv modules for run our hello world app, let's put all the dependencies in a unique cute directory.

I will use `C:\Users\user\Dropbox\AdditionalLibraries\OpenCV`, there I will create a folder for the includes called `include` and a folder for the static libraries called `lib`

Into our `include` I will paste the following folders

- `D:\opencv-master\build\vs_15_2017\opencv2`

- `D:\opencv-master\include\opencv2` 

- `D:\opencv-master\modules\core\include\opencv2`

- `D:\opencv-master\modules\highgui\include\opencv2`

- `D:\opencv-master\modules\imgcodecs\include\opencv2`

- `D:\opencv-master\modules\imgproc\include\opencv2`

Into our `lib` folder I will paste the folder `D:\opencv-master\build\vs_15_2017\lib\Debug`.

Into our `lib` folder I will create a folder called `3rdParty` and inside it I will paste the folder located at: 
`D:\opencv-master\build\vs_15_2017\3rdparty\lib\Debug`

**The folder and file structure should look similar to:**

	- OpenCV\  // $(OPENCV_DIR)
		- include\  // $(OPENCV_DIR)include
			- opencv2\
				+ core\
				+ highgui\
				+ imgcodecs\
				+ imgproc\
				core.hpp
				cvconfig.hpp
				highgui.hpp
				imgcodecs.hpp
				imgproc.hpp
				opencv.hpp
				opencv_modules.hpp
		- lib\  // $(OPENCV_DIR)lib
			- Debug\  // $(OPENCV_DIR)lib\Debug
				opencv_core331d.lib
				opencv_core331d.pdb
				opencv_highgui331d.lib
				opencv_highgui331d.pdb
				opencv_imgcodecs331d.lib
				opencv_imgcodecs331d.pdb
				opencv_imgproc331d.lib
				opencv_imgproc331d.pdb
			- 3rdParty\
				- Debug\ // $(OPENCV_DIR)lib\3rdParty\Debug
					zlibd.lib
					zlibd.pdb
	

### **STEP 16)** Create `OPENCV_DIR` enviroment variable

Tell our visual studio solution where are our OpenCV dependencie let´s creating an enviroment variable pointing to the folder where we grouped our lib and include files.
in cmd write:

	> setx -m OPENCV_DIR C:\Users\user\Dropbox\AdditionalLibraries\OpenCV

### **STEP 17)** Create `opencv_debug` property sheet in VisualStudio

You can enable the `property manager` going to `View > Other Windows > Property Manager` then create a property sheet for debug mode called `opencv_debug`

![enter image description here](https://i.imgur.com/JL1QOeX.png)

### **STEP 18)** Configure`opencv_debug` property sheet

Go the VC++Directories and add the path to your OpenCV `include` and `lib` directories using the previously created `OPENCV_DIR` enviroment variable.

![enter image description here](https://i.imgur.com/KolMCJe.png)

	VC++ Directories
		Include Directories
			$(OPENCV_DIR)include
		Library Directories
			$(OPENCV_DIR)lib\Debug
			$(OPENCV_DIR)lib\3rdParty\Debug
			$(QTDIR)\qtbase\lib

Now let´s tell the `linker` what libraries can `link`at compile-time.

![enter image description here](https://i.imgur.com/qtvVLNa.png)

The next is the exact list of libraries you must to link for run the hello world opencv program (YOU ONLY WILL BE ABLE TO READ `.bmp` IMAGES FOR MORE FUNCTIONALITY READ **STEP 19**).

	opencv_core331d.lib
	opencv_highgui331d.lib
	opencv_imgcodecs331d.lib
	opencv_imgproc331d.lib
	zlibd.lib
	Qt5Cored.lib
	Qt5Guid.lib
	Qt5Testd.lib
	Qt5Widgetsd.lib

### **STEP 19)** More functionality

For get a quasi **complete build of opencv** you could generate the next OpenCV solution:

**Download dependencies:** and paste them into your `opencv/dep` created folder.
	Eigen (Eigen 3.3.4)
		http://eigen.tuxfamily.org/index.php?title=Main_Page
	OpenEXR (openexr-2.2.1.tar.gz)
		http://www.openexr.com/downloads.html

**Check only the following options:** What is not mentioned here should be unchecked (just modify the groups mentioned).

	WITH:
		WITH_1394
		WITH_DIRECTX
		WITH_DSSHOW
		WITH_EIGEN
		WITH_GSTREAMER
		WITH_IPP
		WITH_ITT
		WITH_JASPER
		WITH_JPEG
		WITH_LAPACK
		WITH_OPENEXR
		WITH_OPENMP
		WITH_PNG		
		WITH_QT
		WITH_TIFF
	
	ENABLE:
		ENABLE_PRECOMPILED_HEADERS
		ENABLE_SOLUTION_FOLDERS
		
	BUILD:	
		BUILD_IPP_IW
		BUILD_ITT
		BUILD_JASPER
		BUILD_JPEG
		BUILD_OPENEXR
		BUILD_PNG
		BUILD_TIFF
		BUILD_ZLIB
		BUILD_opencv_apps
		BUILD_opencv_calib3d
		BUILD_opencv_core
		BUILD_opencv_features2d
		BUILD_opencv_flann
		BUILD_opencv_highgui
		BUILD_opencv_imgcodecs
		BUILD_opencv_imgproc
		BUILD_opencv_ml
		BUILD_opencv_objdetect
		BUILD_opencv_photo
		BUILD_opencv_shape
		BUILD_opencv_stitching
		BUILD_opencv_superres
		BUILD_opencv_ts
		BUILD_opencv_video
		BUILD_opencv_videoio
		BUILD_opencv_videostab

# COMMON ERRORS:

**You have more than one visual studio installed:** If you are having a problem of type.

	No CMAKE_C_COMPILER could be found.
	No CMAKE_CXX_COMPILER could be found.

And you are sure you have a c/cpp compilers then you should read this magnific [StackOverflow answer](https://stackoverflow.com/questions/32801638/cmake-error-at-cmakelists-txt30-project-no-cmake-c-compiler-could-be-found).

In my case I had Visual Studio 14 2015 installed and Visual Studio 15 2017, so when I changed the CMake generator from VS2015 to VS2017 the problem was solved.

**can´t read intel ICD OpenGL driver data:** This error is associated to intel graphic drivers, in my case i have to move to older version of installed intel drivers.

My original installed driver was: 21.20.16.4550
My integrated graphic card is an Intel HD Graphics 530

# About blockDim and gridDim

Es conveniente tener bloques o **grids** de **multiples** dimensiones cuando el problema es tambien tiene multiples dimensiones. blockDim nos dice cuantos threas tiene un bloque en la dimension x, y o z mientras que gridDim nos dice cuantos bloques se han instanciado en la direccion x, y, o z.

Una forma fácil de verlo es pensar en los threads como la unidad básica de construcción en CUDA, estos threads se pueden agrupar en bloques y estos bloques se pueden agrupar en grids, el nivel de paralelismo en ultima instancia estará definido por el numero de threads, los bloques son una abstraccion  que nos permite disponer de mecanismos de sincronizan entre los threads que reciden dentro de un bloque. en cambia las dimensiones de los bloques (threads per block) y las "rejillas" de bloques nos proveen una abstracción de alto nivel para simplificar distintos tipos de problemas que requieren soluciones en múltiples "dimensiones".

# Kernel launch parameters

	kernel<<<nBlocks, nThreads>>>();
	// equivalen
	kernel<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>();

Each gGPU can support a limited number of threads per block.

Generally **older GPU** supports **512 Threads per Block**
**New GPU** supports **1024 Threads per Block**

# Is data transfer between CPU and GPU expensive?

Second the CUDA instructor John Owens, you want to minimize data transfer between CPU and GPU as much as you can, if you´re going to move a lot of data and do only a little bit of computation on that data, CUDA or GPU computing probably isn't a great fit for your problem.

# Student notes

The class notes here presented are not of my property except the exercices development and some additional notes added to professor slides, the class notes slides are here just for future fast reference purpouse and were extracted from Intro to Parallel Programming course created by Nvidia©, that is free available at Udacity©.
The main purpouse of this repo is to mantain an organized register of exercises source code and explain core functionality of algorithms involved.