<template>
  <a :class="item.type"
                :href="item.link"
                target="_blank"
                ref="card"
                @mousemove="move"
                @mouseleave="leave"
                @mouseover="over">
                  <div class="reflection" ref="refl"></div>
                  <img :src="item.url"/>
            </a>
</template>

<script>
import Vue from "vue";

export default Vue.extend({
  name: "FrameCard",
  data: () => ({
    debounce: null 
  }),
  props: ["item"],

  methods: {
    over() {
      if (this.item.type == "single-card"){
        return;
      }
      const refl = this.$refs.refl;
      refl.style.opacity = 1;
    },
    leave() {
      if (this.item.type == "single-card"){
        return;
      }
      const card = this.$refs.card;
      const refl = this.$refs.refl;
      card.style.transform = `perspective(500px) scale(1)`;
      refl.style.opacity = 0;
    },

    move() {
      if (this.item.type == "single-card"){
        return;
      }
      const card = this.$refs.card;
      const refl = this.$refs.refl;

      const relX = (event.offsetX + 1) / card.offsetWidth;
      const relY = (event.offsetY + 1) / card.offsetHeight;
      const rotY = `rotateY(${(relX - 0.5) * 60}deg)`;
      const rotX = `rotateX(${(relY - 0.5) * -60}deg)`;
      card.style.transform = `perspective(500px) scale(2) ${rotY} ${rotX}`;

      // const lightX = this.scale(relX, 0, 1, 150, -50);
      // const lightY = this.scale(relY, 0, 1, 30, -100);
      // const lightConstrain = Math.min(Math.max(relY, 0.3), 0.7);
      // const lightOpacity = this.scale(lightConstrain, 0.3, 1, 1, 0) * 255;
      // const lightShade = `rgba(${lightOpacity}, ${lightOpacity}, ${lightOpacity}, 1)`;
      // const lightShadeBlack = `rgba(0, 0, 0, 1)`;
      // refl.style.backgroundImage = `radial-gradient(circle at ${lightX}% ${lightY}%, ${lightShade} 20%, ${lightShadeBlack})`;
    },
    scale: (val, inMin, inMax, outMin, outMax) =>
    outMin + (val - inMin) * (outMax - outMin) / (inMax - inMin) } 
});

</script>


<style scoped>

.frame-content .card {
  background-color: #ccc;
  /* width: 100px;
  height: 100px; */
  transition: all 0.1s ease;
  border-radius: 3px;
  position: relative;
  z-index: 1;
  box-shadow: 0 0 5px rgba(0, 0, 0, 0);
  /* overflow: hidden; */
  cursor: pointer;
}

.frame-content .single-card {
  /* background-color: #ccc; */
  /* width: 100px;
  height: 100px; */
  transition: all 0.1s ease;
  border-radius: 3px;
  position: relative;
  z-index: 1;
  /* box-shadow: 0 0 5px rgba(0, 0, 0, 0); */
  -webkit-transform: scale(1.5);
          transform: scale(1.5);
  /* overflow: hidden; */
  /* cursor: pointer; */
}

.frame-content .card:hover {
  -webkit-transform: scale(2);
          transform: scale(2);
  z-index: 2;
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
}
.frame-content .card:hover img {
  -webkit-filter: grayscale(0);
          filter: grayscale(0);
}
.frame-content .card .reflection {
  position: absolute;
  width: 100%;
  height: 100%;
  z-index: 2;
  left: 0;
  top: 0;
  transition: all 0.1s ease;
  opacity: 0;
  mix-blend-mode: soft-light;
}
.frame-content .card img {
  width: 100%;
  height: 100%;
  -o-object-fit: cover;
     object-fit: cover;
  /* -webkit-filter: grayscale(0.65);
          filter: grayscale(0.65); */
  transition: all 0.3s ease;
}


</style>

