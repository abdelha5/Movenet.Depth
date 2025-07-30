---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Movenet with Depth integration"
  text: "Multi-modal learning in Human Pose Estimation"
  tagline: Project Guide
  image: 
    src: "/movenet.depth/.docs/three_pane_aligned.gif"
    alt: transparent

  actions:
    - theme: alt
      text: documentation
      link: /README.md
    - theme: brand
      text: Test the model
      link: /how-to-run.md

features:
  - title: RGB Data
    details: RGB data refers to visual information captured using the Red, Green, and Blue color channels—the primary colors of light. Each pixel in an RGB image contains a combination of these three color intensities, allowing for the representation of a full-color image as perceived by the human eye. RGB data is the most common form of visual input used in computer vision tasks like object detection, classification, and human pose estimation because it closely resembles how we naturally see and interpret the world.
  - title: Depth data
    details: Depth data provides information about the distance between the camera and objects in a scene. Instead of capturing color, it encodes the spatial structure of the environment by assigning depth values (typically in meters or relative units) to each pixel. This data can be obtained using specialized sensors (like LiDAR or stereo cameras) or estimated from RGB images using models like MiDaS. Depth data is crucial for understanding 3D geometry, enabling applications such as 3D reconstruction, augmented reality, and enhanced pose estimation by providing spatial context to 2D visual inputs.
---

