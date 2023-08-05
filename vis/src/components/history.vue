<template>
    <div class="history-view">
        <div class="topname pl-1">
            <span> History </span>
        </div>
        <div class="history-content">
        </div>
    </div>
</template>

<script>
import Vue from 'vue'
import { mapActions, mapState, mapMutations } from "vuex";
import HistoryRender from "../plugins/render_history";
import * as Global from "../plugins/global";
import * as d3 from "d3";

export default Vue.extend({
  name: 'History',
  data: () => ({}),
  computed: {
    ...mapState([
      "hierarchy_data", "server_url",
      "data_id_mapping", "selected_class",
      "work_history", "in_length_compare", "compare_step"
    ]),
  },
  watch: {
    work_history() {
      let that = this;
      let cur_height = that.history_render.sub_component_update()
      that.svg.attr("height", cur_height);
    }
  },
  methods: {
    ...mapActions(["fetch_step_history"]),
    ...mapMutations(["set_selected_class", "set_in_length_compare", "set_compare_step"]),
  },
  async mounted() {
    window.work_history = this;
    let that = this;
    let container = d3.select(".history-content");
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
      .attr("width", that.layout_width)
      .attr("height", that.layout_height);

    that.history_group = that.svg
      .append("g")
      .attr("id", "history-group")
      .attr(
        "transform",
        `translate(${that.x_margin}, ${that.y_margin})`
      );

    that.history_render = new HistoryRender(that)
  },
})

</script>

<style>
.history-view {
  align-content: baseline;
  max-width: 30%;
}

.history-content {
  background: rgb(255, 255, 255);
  border: 1px solid #c1c1c1;
  border-radius: 5px;
  min-height: 20%;
  max-height: 60%;
  width: 100%;
  padding: 0;
  overflow-x: hidden;
  overflow-y: scroll;
}

</style>