<template>
    <v-col cols="4" class="info-view pl-0 pb-0" style="height:15%">
      <v-col cols="12" class="topname fill-width">
        <div>
          <span> Control </span>
        </div>
      </v-col>
      <v-col cols="12" class="info-content pt-0 fill-height">
        <div
          class="number-slider"
          id="nearest-action-slider"
        >
          <span
            class=""
            v-text="
              '# nearest actions: ' + nearest_action.toFixed(0)
            "
          ></span>
          <v-slider
            v-model="nearest_action"
            max="30"
            :color="'grey'"
            :track-color="'grey lighten-2'"
            :thumb-color="'grey darken-1'"
            style="
                            height: 24px;
                        "
          ></v-slider>
        </div>
        <div
          class="number-slider"
          id="nearest-frame-slider"
        >
          <span
            class=""
            v-text="
              '# nearest frames: ' + nearest_frame.toFixed(0)
            "
          ></span>
          <v-slider
            v-model="nearest_frame"
            max="30"
            :color="'grey'"
            :track-color="'grey lighten-2'"
            :thumb-color="'grey darken-1'"
            style="
                            height: 24px;
                        "
          ></v-slider>
        </div>
        <div
          class="number-slider"
          id="window-size-slider"
        >
          <span
            class=""
            v-text="
              'window size: ' + window_size.toFixed(0)
            "
          ></span>
          <v-slider
            v-model="window_size"
            max="30"
            :color="'grey'"
            :track-color="'grey lighten-2'"
            :thumb-color="'grey darken-1'"
            style="
                            height: 24px;
                        "
          ></v-slider>
        </div>
      </v-col>
      
    </v-col>
</template>

<script>
import Vue from 'vue'

export default Vue.extend({
  name: 'Info',
  data: () => ({}),
  computed:{
    nearest_action: {
      get() {
        return this.$store.state.nearest_action;
      },
      set(value) {
        this.$store.state.nearest_action = value;
      },
    },
    nearest_frame: {
      get() {
        return this.$store.state.nearest_frame;
      },
      set(value) {
        this.$store.state.nearest_frame = value;
      },
    },
    window_size: {
      get() {
        return this.$store.state.window_size;
      },
      set(value) {
        this.$store.state.window_size = value;
      },
    },

  }
})

</script>

<style>
.info-view {
  align-content: baseline
}

.info-content {
  background: rgb(255, 255, 255);
  border: 1px solid #c1c1c1;
  border-radius: 5px;
  height: calc(100% - 32px);
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.number-slider{
  display: flex;
  padding-left: 40px;
  padding-right: 40px;
  padding-top: 5px;
  padding-bottom: 5px;
}
.number-slider span{
  width: 40%; 
  float: left; 
  overflow: hidden; 
  text-overflow: ellipsis;
  font-size: 20px;
}

</style>