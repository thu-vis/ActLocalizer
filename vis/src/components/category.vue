<template>
    <v-col class="cate-view pb-0 fill-height">
      <v-col cols="12" class="topname fill-width pl-1">
        <div>
          <span> Category </span>
        </div>
      </v-col>
      <v-col cols="12" class="cate-content fill-height"></v-col>
    </v-col>
</template>

<script>
import Vue from 'vue'
import { mapActions, mapState, mapMutations } from "vuex";
import ClassRender from "../plugins/render_class";
import * as Global from "../plugins/global";
import * as d3 from "d3";

export default Vue.extend({
  name: 'Category',
  data: () => ({}),
  computed: {
    ...mapState([
      "hierarchy_data", "class_info", "server_url",
      "data_id_mapping", "selected_class"
    ]),
  },
  watch: {
    class_info() {
      console.log("watch class info");
      this.update_component();
    },
    selected_class() {
      this.update_component();
    },
  },
  methods: {
    ...mapActions(["fetch_hierarchy"]),
    ...mapMutations(["set_selected_class"]),
    update_component() {
      console.log("categroy update data");
      this.class_render.sub_component_update();
    }
  },
  async mounted() {
    window.print = console.log;
    window.categroy = this;
    let that = this;
    let container = d3.select(".cate-content");
    let bbox = container.node().getBoundingClientRect();
    that.width = bbox.width;
    that.height = bbox.height;
    that.x_margin = 5;
    that.y_margin = 15;
    that.layout_width = that.width - 2 * that.x_margin;
    that.layout_height = that.height - that.y_margin;

    // animation
    that.create_ani = Global.Animation;
    that.update_ani = Global.Animation;
    that.remove_ani = Global.Animation / 2;

    that.svg = container
      .append("svg")
      .attr("id", "main-svg")
      .attr("width", that.layout_width + 2 * that.x_margin)
      .attr("height", that.layout_height);

    that.class_group = that.svg
      .append("g")
      .attr("id", "class-group")
      .attr(
        "transform",
        `translate(${that.x_margin}, ${that.y_margin})`
      );

    that.class_render = new ClassRender(that);
  },
})

</script>

<style>
.cate-view {
  align-content: baseline;
  max-width: 5.5%;
}

.cate-content {
  background: rgb(255, 255, 255);
  border: 1px solid #c1c1c1;
  border-radius: 5px;
  height: calc(100% - 32px);
  padding: 0;
}


</style>