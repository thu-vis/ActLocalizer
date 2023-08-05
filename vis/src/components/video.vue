<template>
    <v-row class="video-view fill-width mr-0">
      <v-col cols="12" class="topname fill-width"> Video </v-col>
      <v-col cols="12" class="video-content pa-0"></v-col>
    </v-row>
</template>

<script>
import Vue from "vue"
import { mapActions, mapMutations, mapState } from "vuex";
import * as d3 from "d3";
import * as Global from "../plugins/global";
import VideoInfoRender from "../plugins/render_videoinfo";

export default Vue.extend({
  name: "Video",
  components: {
  },
  data: () => ({
    video_layout_ratio: [0.25, 0.4, 0.35],
  }),
  computed: {
    ...mapState(["server_url","selected_action", "selected_video", "key_frames"]),
  },
  watch: {
    async selected_action() {
      let that = this;
      console.log("triger selected action");
      await this.fetch_video_info();
      that.update_data();
    }
  },
  methods: {
    ...mapMutations(["set_selected_class", "set_selected_action"]),
    ...mapActions(["fetch_video_info", "fetch_key_frames"]),
    test(){
      this.set_selected_class(12);
      this.set_selected_action({
        id: "484-36",
        idx: 43758,
        left_bound: 24,
        right_bound: 37,
      });
    },
    async update_data(){
      await this.video_render.sub_component_update(this.selected_video, this.selected_action);
    },
  },
  async mounted() {
    window.video = this;
    window.d3 = d3;
    let that = this;

    let container = d3.select(".video-content");
    console.log("video container", container);
    let bbox = container.node().getBoundingClientRect();
    that.width = bbox.width;
    that.height = bbox.height;
    that.x_margin = 10;
    that.y_margin = 10;
    that.layout_width = that.width - 2 * that.x_margin;
    that.layout_height = that.height - 2 * that.y_margin;
    that.create_ani = Global.Animation;
    that.update_ani = Global.Animation;
    that.remove_ani = Global.Animation / 2;
    
    that.video_group = container
      .append("svg")
      .attr("id", "main-svg")
      .attr("width", that.layout_width)
      .attr("height", that.layout_height)
      .attr("transform", `translate(${that.x_margin}, ${that.y_margin})`);

    that.video_render = new VideoInfoRender(that, this.video_layout_ratio);
  },
})

</script>

<style>
.video-view {
  height: 33%;
  align-content: baseline
}

.video-content {
  background: rgb(255, 255, 255);
  border: 1px solid #c1c1c1;
  border-radius: 5px;
  height: calc(100% - 32px);
}

</style>