<template>
  <v-col class="main-view pl-0 pb-0 fill-height">
    <v-col cols="12" class="topname fill-width pl-1">
      Action
      <div class="corpus-control-panel" style="width: 93%">
        <span>Navigation: </span>
        <div class="control-icon" title="Zoom in" id="zoomin-icon" @click="onZoomInClick()" v-bind:style="{background: this.zoom_in ? '#ccc': 'none'}">
          <svg
            t="1648359130606"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="2344"
            width="19"
            height="17"
            style="display: block"
          >
            <path
              style="fill:rgb(120, 120, 120); opacity:0.9;"
              d="M997.374066 1005.612975a62.540901 62.540901 0 0 1-88.464105 0l-209.512019-209.668371a438.630611 438.630611 0 1 1 90.527954-86.400255l207.604522 207.604521a62.540901 62.540901 0 0 1-0.156352 88.464105zM445.606965 111.278087a328.339731 328.339731 0 1 0 206.384974 583.60042 62.540901 62.540901 0 0 1 13.790268-20.888661 61.321354 61.321354 0 0 1 22.983781-14.165514A327.964486 327.964486 0 0 0 445.606965 111.278087zM570.688767 502.15872h-62.540901v62.540901a62.540901 62.540901 0 0 1-125.081803 0v-62.540901h-62.540901a62.540901 62.540901 0 0 1 0-125.081802h62.540901v-62.540902a62.540901 62.540901 0 0 1 125.081803 0v62.540902h62.540901a62.540901 62.540901 0 0 1 0 125.081802z"
              p-id="2345"
            ></path>
          </svg>
        </div>
        <div class="control-icon" title="Compare actions" id="compare-icon" @click="onCompareClick()" v-bind:style="{background: this.compare_to ? '#ccc': 'none'}">
          <svg
            t="1643350929500"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="2026"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            width="22"
            height="22"
            style="display: block"
          >
            <path
              d="M528 128a48 48 0 0 1 47.776 43.392L576 176V256h272a48 48 0 0 1 47.776 43.392L896 304v544a48 48 0 0 1-43.392 47.776L848 896h-352a48 48 0 0 1-47.776-43.392L448 848V768H176a48 48 0 0 1-47.776-43.392L128 720v-544a48 48 0 0 1 43.392-47.776L176 128h352z m304 192h-256v400a48 48 0 0 1-43.392 47.776L528 768H512v64h320v-224h-114.752l35.904 35.872a32 32 0 0 1-42.24 47.936l-3.04-2.656-90.496-90.528a32 32 0 0 1-2.656-42.24l2.656-3.008 90.496-90.528a32 32 0 0 1 47.936 42.24l-2.656 3.04-35.904 35.84L832 544v-224zM512 192H192v224h173.248l-35.872-35.872a32 32 0 0 1 45.248-45.28l90.528 90.528a32 32 0 0 1 0 45.248l-90.528 90.528a32 32 0 0 1-45.248-45.28L365.248 480H192v224h320V192z"
              p-id="2027"
              style="fill:rgb(120, 120, 120)"
            ></path>
          </svg>
        </div>
        <div class="control-icon" title="Star labeled frames" id="star-icon" @click="onStarIconClick()">
          <svg t="1650439621750" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="27227" width="20" height="20" style="display: block"><path d="M769.097 935.636c-4.457 0-8.93-1.063-13.026-3.216L512 804.104 267.93 932.42a28 28 0 0 1-40.627-29.518l46.614-271.776L76.46 438.653a27.999 27.999 0 0 1 15.519-47.76l272.878-39.651 122.035-247.27a27.998 27.998 0 0 1 50.216 0l122.035 247.27 272.878 39.651a28.002 28.002 0 0 1 15.518 47.76L750.083 631.126l46.613 271.776a28 28 0 0 1-27.599 32.734zM512 744.47c4.476 0 8.951 1.072 13.029 3.216l206.883 108.766-39.511-230.367a27.995 27.995 0 0 1 8.053-24.784l167.37-163.146-231.301-33.61a27.998 27.998 0 0 1-21.082-15.317L512 179.632 408.559 389.227a28 28 0 0 1-21.082 15.317l-231.301 33.61L323.547 601.3a28.004 28.004 0 0 1 8.053 24.784l-39.511 230.367L498.97 747.686A27.99 27.99 0 0 1 512 744.47z" p-id="27228" style="fill:rgb(120, 120, 120); stroke:rgb(120, 120, 120); stroke-width:20;"></path></svg>
        </div>
        <div class="control-icon" title="Show representative frames" id="rep-icon" @click="onRepClick()" v-bind:style="{background: this.change_rep ? '#ccc': 'none'}">
          <svg enable-background="new 0 0 48 48" height="20" id="Layer_1" version="1.1" viewBox="0 0 48 48" width="18" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="display: block; opacity:0.95;"><path clip-rule="evenodd" d="M43,41H5c-2.209,0-4-1.791-4-4V11c0-2.209,1.791-4,4-4h38c2.209,0,4,1.791,4,4v26  C47,39.209,45.209,41,43,41z M45,11c0-1.104-0.896-2-2-2H5c-1.104,0-2,0.896-2,2v26c0,1.104,0.896,2,2,2h38c1.104,0,2-0.896,2-2V11z   M41.334,34.715L35,28.381L31.381,32l3.334,3.334c0.381,0.381,0.381,0.999,0,1.381c-0.382,0.381-1,0.381-1.381,0L19,22.381  L6.666,34.715c-0.381,0.381-0.999,0.381-1.381,0c-0.381-0.382-0.381-1,0-1.381L18.19,20.429c0.032-0.048,0.053-0.101,0.095-0.144  c0.197-0.197,0.457-0.287,0.715-0.281c0.258-0.006,0.518,0.084,0.715,0.281c0.042,0.043,0.062,0.096,0.095,0.144L30,30.619  l4.19-4.19c0.033-0.047,0.053-0.101,0.095-0.144c0.197-0.196,0.457-0.287,0.715-0.281c0.258-0.006,0.518,0.085,0.715,0.281  c0.042,0.043,0.062,0.097,0.095,0.144l6.905,6.905c0.381,0.381,0.381,0.999,0,1.381C42.333,35.096,41.715,35.096,41.334,34.715z   M29,19c-2.209,0-4-1.791-4-4s1.791-4,4-4s4,1.791,4,4S31.209,19,29,19z M29,13c-1.104,0-2,0.896-2,2s0.896,2,2,2s2-0.896,2-2  S30.104,13,29,13z" fill-rule="evenodd" /></svg>
        </div>

        <span>Modification: </span>
        <div class="control-icon" title="Recommend boundary" id="recommend-icon" @click="onRecommendIconClick()">
          <svg t="1625551916203" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2406" width="19" height="19" style="display: block"> <path d="M661.333333 661.397333c-11.157333 2.645333-11.157333 2.645333-12.010666 3.754667l-0.768 0.597333 0.576-0.533333a21.226667 21.226667 0 0 1 12.202666-3.84zM384 725.333333h256v-42.602666c0-7.253333 3.626667-13.653333 9.130667-17.493334 2.816-2.688 5.418667-5.269333 8.768-8.661333 5.546667-5.632 11.946667-12.202667 23.765333-24.448 18.965333-19.626667 23.893333-24.746667 29.568-30.4a282.944 282.944 0 0 0 83.157333-200.768c0-37.290667-7.274667-74.24-21.546666-108.693333a21.333333 21.333333 0 1 1 39.445333-16.32 327.04 327.04 0 0 1 24.768 125.013333c0 86.037333-33.578667 168.874667-95.658667 230.954667-5.397333 5.397333-10.346667 10.496-28.906666 29.717333A3223.168 3223.168 0 0 1 682.666667 692.138667l-0.021334 79.466666c3.2 65.493333-16.448 103.061333-85.034666 103.061334h-174.293334c-66.688 0-89.664-42.88-81.984-103.36v-82.133334c-14.208-10.432-33.834667-29.290667-61.824-57.258666-127.552-127.573333-127.552-334.357333 0-461.909334a325.589333 325.589333 0 0 1 230.954667-95.658666c42.858667 0 85.354667 8.384 125.013333 24.789333a21.333333 21.333333 0 0 1-16.32 39.424 284.373333 284.373333 0 0 0-108.693333-21.546667c-74.837333 0-146.794667 29.184-200.789333 83.178667-110.890667 110.869333-110.890667 290.666667 0 401.536 13.184 13.184 28.608 27.989333 40.938666 39.210667 6.656 6.08 12.224 10.944 16.298667 14.229333 1.92 1.536 3.370667 2.645333 4.245333 3.2-2.069333-1.344-3.285333-1.941333-8.490666-1.941333a21.333333 21.333333 0 0 1 21.333333 21.333333V725.333333z m0 42.666667v4.693333l-0.192 2.837334c-5.248 38.997333 3.904 56.469333 39.509333 56.469333h174.293334c37.333333 0 44.586667-13.866667 42.389333-59.349333V768H384z m0 128h256v42.666667H384v-42.666667z" style="fill:rgb(120, 120, 120); stroke:rgb(120, 120, 120); stroke-width:35;" p-id="4058"></path></svg>        
        </div>
        <div class="control-icon" title="Change current label" id="remove-icon" @click="onRemoveIconClick()">
          <svg t="1653629643782" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="6228" width="18" height="18" style="display: block"><path d="M920.528912 444.49616l-340.111125-340.114737a135.932464 135.932464 0 0 0-96.155899-39.758503L200.418112 64.662657h-0.018063C125.275769 64.662657 64.301413 125.40943 64.301413 200.548159v283.673992a136.091412 136.091412 0 0 0 39.881326 96.199248l340.230335 340.241173a136.022775 136.022775 0 0 0 192.369598 0l283.74624-283.76069c53.128139-53.131752 53.128139-139.277583 0-192.405722z m-45.209673 147.192436l-283.746241 283.760691a72.064544 72.064544 0 0 1-101.924964 0l-340.212273-340.241173A72.187367 72.187367 0 0 1 128.241582 484.218538V200.548159A72.021195 72.021195 0 0 1 200.396437 128.602826l283.843776-0.039737a71.887535 71.887535 0 0 1 50.935389 21.053294l340.114737 340.1039a72.104281 72.104281 0 0 1 0.0289 101.968313z" fill="#787878" p-id="6229"></path><path d="M384.002258 255.999097c-70.69543 0-128.003161 57.307731-128.003161 128.003161S313.306828 511.998194 384.002258 511.998194s127.999548-57.307731 127.999548-127.999549S454.690463 255.999097 384.002258 255.999097z m45.296371 173.29592a64.059379 64.059379 0 1 1 18.759396-45.292759 63.640337 63.640337 0 0 1-18.763008 45.292759z" fill="#787878" p-id="6230"></path></svg>
        </div>
        <div class="control-icon" title="Update" id="update-icon" @click="onUpdateIconClick()">
          <svg t="1650438931300" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="16808" width="18" height="18" style="display: block"><path d="M778.154667 665.6H704a38.4 38.4 0 1 1 0-76.8h122.624a38.229333 38.229333 0 0 1 23.978667 0h6.997333a38.4 38.4 0 0 1 38.4 38.4v153.6a38.4 38.4 0 1 1-76.8 0v-38.272a384.128 384.128 0 0 1-682.410667-148.522667 38.4 38.4 0 0 1 75.050667-16.341333A307.328 307.328 0 0 0 778.154667 665.6zM245.888 358.4H320a38.4 38.4 0 0 1 0 76.8H166.4a38.4 38.4 0 0 1-38.4-38.4V243.2a38.4 38.4 0 0 1 76.8 0v38.357333a384.128 384.128 0 0 1 681.898667 146.090667 38.4 38.4 0 1 1-74.922667 16.810667A307.328 307.328 0 0 0 245.888 358.4z" style="fill:rgb(120, 120, 120)" p-id="16809"></path></svg>
        </div>
        <div class="control-icon" title="History" id="history-icon" @click="onHistoryIconClick()">
          <svg t="1650439358147" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="22670" width="18" height="18" style="display: block"><path d="M511.73808594 93.19941406c-140.67070313 0-264.92695313 69.1453125-341.01035156 175.08955078v-70.47597656c0-21.66767578-17.61679688-39.22998047-39.346875-39.22998047s-39.346875 17.56318359-39.346875 39.22998047v156.91992188c0 21.66240234 17.61679688 39.22998047 39.346875 39.22998047h170.50341796c21.73007813 0 39.346875-17.56669922 39.346875-39.22998047 0-21.66767578-17.61679688-39.22998047-39.346875-39.22998047h-68.4421875c61.73085937-86.98798828 163.28496094-143.84267578 278.296875-143.84267578 188.32851562 0 341.00507812 152.22128906 341.00507813 339.99345703 0 187.76865234-152.6765625 339.99345703-341.00507813 339.99345703-158.73574219 0-291.74589844-108.29179688-329.75771484-254.70439453l-0.68291016 0.21269531c-5.3015625-15.41162109-19.54248047-26.65986328-36.8024414-26.65986328-21.73007813 0-39.346875 17.56318359-39.346875 39.22558594 0 3.86630859 1.16367187 7.36875 2.20605469 10.89316406l-0.40957032 0.12919922c0.25839844 0.95009766 0.73212891 1.8 0.99580078 2.74482422 0.28916016 0.76201172 0.48164063 1.52314453 0.81650391 2.2640625 49.92539062 175.47011719 210.98935547 304.35380859 402.98378906 304.35380859 231.79394531 0 419.69970703-187.35029297 419.69970703-418.45341797-0.00175781-231.10751953-187.90839844-418.45429688-419.70322265-418.45429687z m0 176.53183594c-21.73007813 0-39.346875 17.56318359-39.346875 39.22998047v235.37988281c0 21.66767578 17.61679688 39.22998047 39.346875 39.22998047h170.50253906c21.73007813 0 39.346875-17.56318359 39.346875-39.22558594 0-21.67119141-17.61679688-39.22998047-39.346875-39.22998047H551.08496094v-196.15429687c0-21.66767578-17.61767578-39.22998047-39.34775391-39.22998047z" p-id="22671" style="fill:rgb(120, 120, 120)"></path></svg>
        </div>
        
        <span>Tour function: </span>
        <div class="control-icon help"
          title="Get help"
          id="help-icon"
          @click="onHelpIconClick()"
        >
          <div class="question">?</div>
        </div>
      </div>
    </v-col>
    <v-col cols="12" class="corpus-content pa-0">
      <video id="selected-action-video" controls>
      </video>
      <History class="history-view-in-corpus"
        v-bind:style="{visibility: this.show_history ? 'visible': 'hidden'}"/>
      <Labels class="labels-view-in-corpus"
        v-bind:style="{visibility: this.show_labels ? 'visible': 'hidden'}"/>
    </v-col>
  </v-col>
</template>

<script>
import { mapActions, mapState, mapMutations } from "vuex";
// import MatrixRender from "../plugins/render_matrix";
import ClassRender from "../plugins/render_class";
// import ScatterPlotRender from "../plugins/render_scatterplot";
import Vue from "vue";
import * as Global from "../plugins/global";
import * as d3 from "d3";
import History from "../components/history.vue"
import Labels from "../components/labels.vue"
import AlignmentsRender from "../plugins/render_alignments";

export default Vue.extend({
  name: "Corpus",
  data: () => ({
    zoom_in: false,
    compare_to: false,
    change_rep: false,
    last_boundary_id: -1,
    bound_pos: 0,
    show_history: false,
    show_labels: false,
  }),
  components: {
    History,
    Labels,
  },
  computed: {
    ...mapState([
      "hierarchy_data",
      "class_info",
      "server_url",
      "data_id_mapping",
      "selected_class",
      "selected_ids",
      "selected_details",
      "expand_node_ids",
      "selected_cluster",
      "anchor_alignments",
      "history",
      "in_length_compare",
      "compare_step"
    ]),
  },
  watch: {
    hierarchy_data() {
      console.log("watch hierarchy_data");
      this.update_component();
      Global.end_loading(Global.Animation);
      this.show_labels = false;
    },
    in_length_compare(is_in_compare) {
      this.update_component();
      if(is_in_compare) this.show_history = false;
    },
    selected_details(){
      console.log("watch selected_details");
    },
    selected_class() {
      console.log("watch selected class");
      Global.begin_loading();
      this.fetch_hierarchy({ id: this.selected_class });
      this.set_history([]);
    },
    selected_cluster() {
      console.log("watch selected cluster", this.selected_cluster);
      Global.begin_loading();
      this.fetch_anchor_alignments({
        class: this.selected_class,
        id: this.selected_cluster.selected_action,
      });
      this.push_history({
        id: this.selected_cluster.selected_action,
        is_comparison: true})
    },
    anchor_alignments() {
      console.log("anchor", this.anchor_alignments);
      this.aligns_render.data = this.anchor_alignments;
      this.aligns_render.anchor_alignments = this.anchor_alignments;
      this.last_boundary_id = -1;
      // console.log(this.aligns_render)
      // this.aligns_render.stack.push(this.aligns_render.data)

      this.aligns_render.sub_component_view_update();
      Global.end_loading(Global.Animation);
    },
    nearest_action() {
      this.set_corpus_arguments({ arg_id: 0, value: this.nearest_action });
    },
    nearest_frame() {
      this.set_corpus_arguments({ arg_id: 1, value: this.nearest_frame });
    },
    window_size() {
      this.set_corpus_arguments({ arg_id: 2, value: this.window_size });
    },
  },
  methods: {
    ...mapActions(["fetch_hierarchy", "fetch_key_frames_directly", 
                  "fetch_anchor_alignments", "set_corpus_arguments",
                  "fetch_pred_scores_of_video_with_given_boundary",
                  "fetch_update", "fetch_recommendation", "remove_action",
                  "fetch_work_history"]),
    ...mapMutations([
      "set_hierarchy", "set_selected_ids", "set_selected_cluster",
      "set_expand_node_ids", "set_selected_class", "set_constraint",
      "set_history", "push_history", "pop_history", "set_in_length_compare"]),
    test() {
      return
    },
    update_component() {
      console.log("corpus update data");
      let that = this;
      this.hierarchy = this.hierarchy_data;
      that.aligns_render.sub_component_update();
    },
    onZoomInClick() {
      console.log("click Zoom in");
      this.zoom_in = !this.zoom_in;
      if(this.zoom_in) {
        this.compare_to = false;
        this.change_rep = false;
      }
    },
    onCompareClick() {
      console.log("click compare");
      this.compare_to = !this.compare_to;
      if(this.compare_to){
        this.zoom_in = false;
        this.change_rep = false;
      }
    },
    onStarIconClick() {
      console.log("click star");
      this.aligns_render.star_labeled_frames();
    },
    onRepClick(){
      console.log("click Change rep image");
      this.change_rep = !this.change_rep;
      if(this.change_rep) {
        this.zoom_in = false;
        this.compare_to = false;
      }
    },
    onRecommendIconClick() {
      console.log("click recommend");
      let that = this;
      if(!this.aligns_render.cards[0].current_action || this.last_boundary_id === -1) return;
      this.fetch_recommendation([this.aligns_render.cards[0].current_action.rep_action, this.last_boundary_id, this.bound_pos])
      .then(resp => {
        that.aligns_render.recommend_update(resp);
      })
    },
    onRemoveIconClick() {
      console.log("click remove")
      this.show_history = false;
      this.show_labels = !this.show_labels;
    },
    onUpdateIconClick() {
      console.log("click update", this.aligns_render.operation);
      this.set_constraint(this.aligns_render.operation);
      this.fetch_update();
      this.show_labels = false;
      this.show_history = false;
    },
    onHistoryIconClick() {
      console.log("click history");
      this.show_history = !this.show_history;
      this.show_labels = false;      
      if(this.show_history) this.fetch_work_history();
      // if(this.in_length_compare) this.set_in_length_compare(false);
    },
    onHelpIconClick() {
      console.log("click help");
    },
  },
  async mounted() {
    window.print = console.log;
    window.corpus = this;
    Global.begin_loading();
    let that = this;
    let container = d3.select(".corpus-content");
    let bbox = container.node().getBoundingClientRect();
    let scale = Math.max(1, Global.logic_height / bbox.height)
    that.width = bbox.width * scale;
    that.height = bbox.height * scale;
    that.x_margin = 10;
    that.y_margin = 30;
    that.layout_width = that.width - 2 * that.x_margin
    that.layout_height = that.height - 2 * that.y_margin
    that.aligns_height = that.layout_height;
    that.aligns_width = that.layout_width;

    // animation
    that.create_ani = Global.Animation;
    that.update_ani = Global.Animation;
    that.remove_ani = Global.Animation / 2;

    that.svg = container
      .append("svg")
      .attr("id", "main-svg")
      .attr("width", bbox.width - that.x_margin)
      .attr("height", bbox.height)
      .attr("viewBox", `0 0 ${that.layout_width} ${that.layout_height}`)

    let defs = that.svg.append("defs");
    defs.append("radialGradient").attr("id", "middle-uncertain");

    that.aligns_group = that.svg
      .append("g")
      .attr("id", "alignment-group")
      .attr("transform", `translate(${that.x_margin}, ${that.y_margin})`);

    that.aligns_render = new AlignmentsRender(that);
  },
});
</script>

<style>
html {
  font-size: 0.835vw;
}

.corpus-content {
  background: rgb(255, 255, 255);
  border: 1px solid #c1c1c1;
  border-radius: 5px;
  height: calc(100% - 32px);
}

.corpus-content video {
  position: absolute;
  right: 100px;
  bottom: 55px;
  width: 400px;
  height: 270px;
  visibility:hidden;
  object-fit:fill;
}

.corpus-control-panel {
  display: flex;
  align-items: center;
  white-space: nowrap;
  text-overflow: ellipsis;
  font-size: 0.835vw;
  font-weight: 400;
  justify-content: flex-end;
}

.corpus-control-panel span {
  padding-left: 32px;
  padding-right: 7px;
}

.number-slider {
  display: flex;
  width: 32%;
  padding-left: 25px;
  padding-right: 25px;
  padding-top: 5px;
  padding-bottom: 5px;
}

.number-slider span {
  width: 40%;
  float: left;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 17px;
}

.control-icon {
  cursor: pointer;
  margin-left: 5px;
  margin-right: 5px;
}

.control-icon:hover {
  background: #ccc;
}

.help .question {
  height: 20px;
  width: 20px;
  background: #ccc;
  font-size: 16px;
  line-height: 20px;
  text-align: center;
  border-radius: 50%;
  cursor: pointer;
}

#help-icon:hover {
  background: #ddd;
}

.topname {
  display: flex;
  align-items: center;
  font-size: 18px;
  font-family: "Roboto", "Helvetica", "Arial", sans-serif;
  font-weight: 600;
  background: rgb(238, 238, 238);
  border-radius: 5px;
  padding-left: 5px;
  color: rgb(120, 120, 120);
  height: 22px;
  justify-content: space-between;
}

.class-name {
  cursor: default;
}

.node-name {
  cursor: default;
}

.title-text {
  font-size: 18px;
  stroke: rgb(114, 114, 114);
  fill: rgb(114, 114, 114);
}
/* 
.seg-rect{
  fill: rgb(220, 220, 220);
} */
path.expand-path {
  pointer-events: none;
}

.history-view-in-corpus {
  position: absolute;
  right: 20px;
  top: 45px;
  height: 100%;
  width: 24%;
}

.labels-view-in-corpus {
  position: absolute;
  right: 25px;
  top: 45px;
  height: 100%;
  width: 15%;
}
</style>
