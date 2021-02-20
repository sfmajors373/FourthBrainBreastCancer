<template>
  <div class="hello">
    <h1>Cancer Classification</h1>
    <h3>A portion of CancerMap</h3>
    <input type="file" @change="onFileChanged">
    <button @click="onUpload">Upload!</button>
    <div v-if="conclusion.script">
      <p>{{conclusion.script}}</p>
    </div>
    <p>Please upload a jpg tile from the patient slide</p>
    <div v-if="items.image">
      <p>Original Image</p>
      <img :src="items.image" alt="Original Image">
    </div>
  </div>
</template>

<script>
const axios = require('axios');

export default {
  name: 'HelloWorld',
  props: {
    msg: String,
  },
  data() {
    return {
      selectedFile: null,
      items: {
        image: null,
      },
      conclusion: {
        script: null,
      },
    };
  },
  methods: {
    onFileChanged(event) {
      // eslint-disable-next-line
      this.selectedFile = event.target.files[0];
      // console.log('Files: ', event.target.files);
      // eslint-disable-next-line
      this.createImage(event, event.target.files[0]);
      // console.log(this.selectedFile);
    },
    onUpload() {
      // upload file, get it from this.selectedFile
      const formData = new FormData();
      // formData.append('myFile', this.selectedFile, this.selectedFile.name);
      formData.myFile = this.selectedFile;
      console.log('Form Data: ', formData);
      axios.post('my-domain.com/file-upload', formData, {
        // onUploadProgress: (progressEvent) => {
        // console.log(progressEvent.loaded / progressEvent.total);
        // },
      });
    },
    createImage(event, file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        this.items.image = e.target.result;
      };
      reader.readAsDataURL(file);
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 40px;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
img {
  height: auto;
  width: 50%;
}
</style>
