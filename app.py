import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

class ImageFilter:
    @staticmethod
    def average_blur(image, kernel_size):
        return cv2.blur(image, (kernel_size, kernel_size))

    @staticmethod
    def gaussian_blur(image, kernel_size):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def median_blur(image, kernel_size):
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def bilateral_blur(image, diameter, sigmaColor, sigmaSpace):
        return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)

    @staticmethod
    def box_filter(image, kernel_size):
        return cv2.boxFilter(image, -1, (kernel_size, kernel_size))

    @staticmethod
    def radial_blur(image, amount):
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (center_x, center_y), min(h, w) // 2, 1, -1)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=amount, sigmaY=amount)
        result = image.copy().astype(np.float32)
        result = result * (1 - mask[:,:,np.newaxis]) + blurred * mask[:,:,np.newaxis]
        return result.astype(np.uint8)

    @staticmethod
    def sharpen_filter(image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def edge_detection(image):
        return cv2.Canny(image, 100, 200)

    @staticmethod
    def emboss_filter(image):
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def sketch_filter(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        gauss_gray = cv2.GaussianBlur(inv_gray, (21,21), 0)
        sketch = cv2.divide(gray, 255 - gauss_gray, scale=256.0)
        return sketch

def main():
    st.title("Blur Image Filter Processing System")
    
    st.sidebar.header("Image Filter Options")
    
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_image = np.array(image)
        
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)
        
        filter_category = st.sidebar.selectbox(
            "Select Filter Category",
            [
                "Blur Filters", 
                "Edge & Sketch Filters", 
                "Texture Filters"
            ]
        )
        
        if filter_category == "Blur Filters":
            filter_type = st.sidebar.selectbox(
                "Select Blur Filter",
                [
                    "Average Blur", 
                    "Gaussian Blur", 
                    "Median Blur", 
                    "Bilateral Filter", 
                    "Box Filter",
                    "Radial Blur"
                ]
            )
            
            if filter_type == "Average Blur":
                kernel_size = st.sidebar.slider("Kernel Size", 1, 21, 5, step=2)
                processed_image = ImageFilter.average_blur(original_image, kernel_size)
            
            elif filter_type == "Gaussian Blur":
                kernel_size = st.sidebar.slider("Kernel Size", 1, 21, 5, step=2)
                processed_image = ImageFilter.gaussian_blur(original_image, kernel_size)
            
            elif filter_type == "Median Blur":
                kernel_size = st.sidebar.slider("Kernel Size", 1, 21, 5, step=2)
                processed_image = ImageFilter.median_blur(original_image, kernel_size)
            
            elif filter_type == "Bilateral Filter":
                diameter = st.sidebar.slider("Diameter", 1, 21, 9)
                sigmaColor = st.sidebar.slider("Sigma Color", 1, 255, 75)
                sigmaSpace = st.sidebar.slider("Sigma Space", 1, 255, 75)
                processed_image = ImageFilter.bilateral_blur(original_image, diameter, sigmaColor, sigmaSpace)
            
            elif filter_type == "Box Filter":
                kernel_size = st.sidebar.slider("Kernel Size", 1, 21, 5, step=2)
                processed_image = ImageFilter.box_filter(original_image, kernel_size)
            
            elif filter_type == "Radial Blur":
                amount = st.sidebar.slider("Blur Amount", 1, 50, 10)
                processed_image = ImageFilter.radial_blur(original_image, amount)
        
        elif filter_category == "Edge & Sketch Filters":
            filter_type = st.sidebar.selectbox(
                "Select Edge/Sketch Filter",
                [
                    "Edge Detection", 
                    "Sketch Filter"
                ]
            )
            
            if filter_type == "Edge Detection":
                processed_image = ImageFilter.edge_detection(original_image)
            
            elif filter_type == "Sketch Filter":
                processed_image = ImageFilter.sketch_filter(original_image)
        
        elif filter_category == "Texture Filters":
            filter_type = st.sidebar.selectbox(
                "Select Texture Filter",
                [
                    "Sharpen", 
                    "Emboss"
                ]
            )
            
            if filter_type == "Sharpen":
                processed_image = ImageFilter.sharpen_filter(original_image)
            
            elif filter_type == "Emboss":
                processed_image = ImageFilter.emboss_filter(original_image)
        
        st.subheader(f"Processed Image - {filter_type}")
        st.image(processed_image, use_container_width=True)
        
        # Download button
        pil_image = Image.fromarray(processed_image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = buffered.getvalue()
        
        st.download_button(
            label="Download Processed Image",
            data=img_str,
            file_name="processed_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()