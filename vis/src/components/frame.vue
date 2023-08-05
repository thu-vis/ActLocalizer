<template>
  <v-row class="frame-view fill-width mr-0">
    <v-col cols="12" class="topname fill-width">
      
      <div>
        <span> Frame </span>
      </div>
      <div class="frame-control-panel" style="width: 90%">
        <div id="left-page" @click="LeftPage()">
          <svg
            t="1626190600743"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="1177"
            width="20"
            height="20"
          >
            <path
              d="M495.976 476.195c19.777 17.656 21.494 48 3.837 67.774a48.003 48.003 0 0 1-3.837 3.836L536.082 512l-40.106-35.805zM864 212.083v-82.217a8 8 0 0 0-13.328-5.967L442.69 488.13c-0.9 0.804-1.754 1.657-2.558 2.557-11.772 13.184-10.626 33.412 2.558 45.183l407.983 364.231A8 8 0 0 0 864 894.134v-82.217a16 16 0 0 0-5.344-11.936L536.082 512l322.574-287.981A16 16 0 0 0 864 212.083zM495.976 476.195c19.777 17.656 21.494 48 3.837 67.774a48.003 48.003 0 0 1-3.837 3.836L536.082 512l-40.106-35.805zM864 212.083v-82.217a8 8 0 0 0-13.328-5.967L442.69 488.13c-0.9 0.804-1.754 1.657-2.558 2.557-11.772 13.184-10.626 33.412 2.558 45.183l407.983 364.231A8 8 0 0 0 864 894.134v-82.217a16 16 0 0 0-5.344-11.936L536.082 512l322.574-287.981A16 16 0 0 0 864 212.083z"
              p-id="1178"
            ></path>
            <path
              d="M223.976 476.195c19.777 17.656 21.494 48 3.837 67.774a48.003 48.003 0 0 1-3.837 3.836L264.082 512l-40.106-35.805zM592 212.083v-82.217a8 8 0 0 0-13.328-5.967L170.69 488.13c-0.9 0.804-1.754 1.657-2.558 2.557-11.772 13.184-10.626 33.412 2.558 45.183l407.983 364.231A8 8 0 0 0 592 894.134v-82.217a16 16 0 0 0-5.344-11.936L264.082 512l322.574-287.981A16 16 0 0 0 592 212.083zM223.976 476.195c19.777 17.656 21.494 48 3.837 67.774a48.003 48.003 0 0 1-3.837 3.836L264.082 512l-40.106-35.805zM592 212.083v-82.217a8 8 0 0 0-13.328-5.967L170.69 488.13c-0.9 0.804-1.754 1.657-2.558 2.557-11.772 13.184-10.626 33.412 2.558 45.183l407.983 364.231A8 8 0 0 0 592 894.134v-82.217a16 16 0 0 0-5.344-11.936L264.082 512l322.574-287.981A16 16 0 0 0 592 212.083z"
              p-id="1179"
            ></path>
          </svg>
        </div>
        <div id="page-num">
          <span class="" v-text=" (grid_page+1) + '/' + total_page"
            style="overflow: hidden; text-overflow: ellipsis;white-space: nowrap;"></span>
        </div>

        <div id="right-page" @click="RightPage()">
          <svg
            t="1626190628077"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="1968"
            width="20"
            height="20"
          >
            <path
              d="M160.117 212.026v-82.233a8 8 0 0 1 13.33-5.966l407.697 364.298c0.9 0.804 1.753 1.658 2.556 2.558 11.764 13.186 10.62 33.419-2.556 45.192L173.448 900.173a8 8 0 0 1-13.33-5.966v-82.233a16 16 0 0 1 5.338-11.93L487.814 512 165.456 223.957a16 16 0 0 1-5.339-11.931z m272.057 0v-82.233a8 8 0 0 1 13.33-5.966l407.697 364.298c0.9 0.804 1.753 1.658 2.556 2.558 11.764 13.186 10.62 33.419-2.556 45.192L445.505 900.173a8 8 0 0 1-13.33-5.966v-82.233a16 16 0 0 1 5.339-11.93L759.87 512 437.514 223.957a16 16 0 0 1-5.34-11.931z"
              p-id="1969"
            ></path>
          </svg>
        </div>
      </div>
    </v-col>
    <v-col cols="12" class="frame-content pa-8">
      <FrameCard
        v-for="item in grid_data"
        :key="item.id"
        :item="item"
      ></FrameCard>
    </v-col>
  </v-row>
</template>

<script>
import Vue from "vue";
import { mapActions, mapMutations, mapState } from "vuex";
import * as d3 from "d3";
// import * as d3ContextMenu from "d3-context-menu";
import * as Global from "../plugins/global";
import FrameCard from "../components/framecard.vue";

export default Vue.extend({
  name: "Frame",
  components: {
    FrameCard,
  },
  data: () => ({
    mode: "unselected",
    grid_page: 0,
    grid_data: [],
    single_data: [],
    test_data: [],
    total_page: 0,
  }),
  computed: {
    ...mapState(["server_url", "selected_frames"]),
  },

  watch: {
    selected_frames() {
      let that = this;
      console.log("triger selected frame");
      that.grid_page = 0;
      that.update_data();
    },
  },

  methods: {
    // ...mapActions(["fetch_hierarchy"]),
    ...mapMutations(["set_selected_frames"]),
    test() {
      // this.set_selected_frames(["0-1"]);
      this.set_selected_frames([
        "0-1",
        "0-2",
        "0-3",
        "0-4",
        "0-5",
        "0-6",
        "0-7",
        "0-8",
        "0-9",
        "0-10",
        "0-11",
        "0-12",
        "0-13",
        "0-14",
        "0-15",
        "0-16",
        "0-17",
        "0-18",
        "0-19",
        "0-20",
        "0-21",
        "0-22",
        "0-15",
        "0-16",
        "0-17",
        "0-18",
        "0-19",
        "0-20",
        "0-21",
        "0-22",
        "0-15",
        "0-16",
        "0-17",
        "0-18",
        "0-19",
        "0-20",
        "0-21",
        "0-22",
        "0-15",
        "0-16",
        "0-17",
        "0-18",
        "0-19",
        "0-20",
        "0-21",
        "0-22",
        "0-15",
        "0-16",
        "0-17",
        "0-18",
        "0-19",
        "0-20",
        "0-21",
        "0-22",
        "0-15",
        "0-16",
        "0-17",
        "0-18",
        "0-19",
        "0-20",
        "0-21",
        "0-22",
        "0-15",
        "0-16",
        "0-17",
        "0-18",
        "0-19",
        "0-20",
        "0-21",
        "0-22",
      ]);
    },
    test_single() {
      this.set_selected_frames(["0-1"]);
    },
    LeftPage(){
      console.log("left page");
      this.change_direction(-1);
    },
    RightPage(){
      console.log("right page");
      this.change_direction(1);
    },
    update_data() {
      let that = this;
      console.log("update_data");
      if (that.selected_frames.length > 1) {
        // get data by selected_node
        that.show_turn_page_icon(true);
        d3.selectAll(".frame-content").style(
          "grid-template-columns",
          "repeat(auto-fill, 120px)"
        );
        that.mode = "grid";
        let data = that.selected_frames;
        let num_in_one_page = 12;
        that.total_page = Math.ceil(data.length / num_in_one_page);
        let in_page = function (i) {
          if (
            i >= that.grid_page * num_in_one_page &&
            i < (that.grid_page + 1) * num_in_one_page
          ) {
            console.log(i, that.grid_page, num_in_one_page);
            return true;
          } else {
            return false;
          }
        };
        data = data.map((d, i) =>
          Object({
            idx: d,
            url: `${this.server_url}/Frame/single_frame?filename=${d}`,
            type: "card",
            show: in_page(i),
          })
        );
        // that.grid_data = data;
        that.grid_data = data.filter((d) => d.show);
      } else if (that.selected_frames.length == 1) {
        // one-frame mode, focus frame
        that.show_turn_page_icon(false);
        d3.selectAll(".frame-content").style("grid-template-columns", "none");
        that.mode = "single";
        that.grid_data = [
          {
            url: `${this.server_url}/Frame/single_frame?filename=${that.selected_frames[0]}`,
            idx: that.selected_frames[0],
            type: "single-card",
          },
        ];
      } else {
        that.grid_data = [];
      }
    },
    change_direction(pace){
      console.log("change_direction");
      let that = this;
      let next = that.grid_page + pace;
      if (next < 0 || next >= that.total_page) return;
      that.grid_page = next;
      that.update_data()
    },
    show_turn_page_icon(flag){
      if (flag){
        d3.select("#left-page").style("display", "block");
        d3.select("#right-page").style("display", "block");
        d3.select("#page-num").style("display", "block");
      }
      else{
        d3.select("#left-page").style("display", "none");
        d3.select("#right-page").style("display", "none");
        d3.select("#page-num").style("display", "none");
      }
    }
  },

  async mounted() {
    window.frame = this;
    window.d3 = d3;
    let that = this;
    that.show_turn_page_icon(false);
    // let container = d3.select(".frame-content");
    // console.log("frame container", container);
    // let bbox = container.node().getBoundingClientRect();
    // that.width = bbox.width;
    // that.height = bbox.height;
    // that.x_margin = 10;
    // that.y_margin = 10;
    // that.layout_width = that.width - 2 * that.x_margin;
    // that.layout_height = that.height - 2 * that.y_margin;
  },
});
</script>


<style>
.frame-view {
  height: 33%;
  align-content: baseline;
}

.frame-content {
  display: grid;
  grid-template-columns: repeat(auto-fill, 120px);
  grid-column-gap: 30px;
  grid-row-gap: 30px;
  align-items: center;
  justify-content: center;
  width: 100%;
  background: rgb(255, 255, 255);
  border: 1px solid #c1c1c1;
  border-radius: 5px;
  height: calc(100% - 32px);
}

.topname {
  display: flex;
  align-items: center;
  font-size: 20px;
  font-family: "Roboto", "Helvetica", "Arial", sans-serif;
  font-weight: 600;
  background: rgb(238, 238, 238);
  border-radius: 5px;
  padding-left: 10px;
  color: rgb(120, 120, 120);
  height: 22px;
  justify-content: space-between;
}

.frame-control-panel {
  /* justify-content: flex-end; */
  display: flex;
  align-items: center;
  white-space: nowrap;
  text-overflow: ellipsis;
  /* font-size: 16px; */
  font-size: 0.835vw;
  font-weight: 400;
}

#left-page:hover {
  background: #ddd;
}

#right-page:hover {
  background: #ddd;
}
</style>

