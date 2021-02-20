<template>
  <div class="about">
    <h1>This is an about page</h1>
    <div id="link1">
      <a href="https://github.com/sfmajors373/FourthBrainBreastCancer">Github</a>
    </div>
    <div id="link2">
      <a href="">Documentation</a>
    </div>
    <h5>Cancer Classification Overview</h5>
    <!-- eslint-disable -->
    <p>Cancer Classification is one portion of the CancerMap project.  It is intended to mitigate some of the potential ethical concerns with patients being hesitant to fully trust AI with their health.  Loading a tile of the whole slide will use the model to predict whether there are cancer cells present in the tile.  The API will also return images using GradCam which show where the activations are highest within the image to showcase to a patient the areas of the image that the algorithm focused on.  Hopefully, this will provide patients with a bit of explainability to sooth their concerns of processes they do not understand.</p>
    <h5>Future Work</h5>
    <p>Future work will include combining the classification portion of CancerMap with the segmentation portion of CancerMap so that the image only needs uploaded once to receive all information.  Additionally, the project should be able to handle uploads of entire Whole Slide Images.  The end result should also stitch back togehther all the tiles from the slide and provide comprehensive answers about the cancer status of the entire slide.  Another point that would be nice to address in the future is making the app lightweight enough to be downloaded onto a pathologist's computeer and run there.  This would prevent the concerns with sending patient data into the cloud and eliminate concerns with the data being leaked or intercepted by a bad actor.</p>
  </div>
</template>
