**Image Processing App**
This Python application allows users to perform various image processing tasks using a graphical user interface (GUI) built with Tkinter. Users can select an image file, apply different image processing techniques, and visualize the results in real-time.  

To run the application, follow these steps:  

Ensure you have Python installed on your system.  
_Clone this repository to your local machine or download the source code files_  

Install the required Python libraries by running the following command:  
_**pip install numpy**_  

Run the main script image_processing_app.py using the following command:  
_**python image_processing_app.py**_  

**Features:**  
Supported Image Processing Techniques:  
-Linear Mapping: Adjusts the contrast of an image using linear mapping techniques.  
-Nonlinear Mapping: Applies nonlinear transformations to enhance image contrast.  
-Histogram Compression: Compresses the dynamic range of an image histogram.  
-Histogram Stretching: Stretches the dynamic range of an image histogram to improve contrast.  
-Digital Negative: Generates a digital negative of the input image.  
-ACE Filter: Applies Adaptive Contrast Enhancement (ACE) filtering to improve image contrast.  
-Histogram Specification: Matches the histogram of the input image to a desired histogram.  
-Pseudomedian Filter: Removes noise from an image using a pseudomedian filtering algorithm.  
-MMSE Filter: Applies Minimum Mean Squared Error (MMSE) filtering to denoise an image.  

**User Interface:**  
Users can select an image file from their local filesystem.  
The GUI provides buttons for each image processing technique, allowing users to apply them interactively.  
Real-time visualization of original and processed images is provided within the application window.  

**Dependencies:**  
Python 3.x  
NumPy: For numerical operations on image data.  
