/* eslint-disable */

import { cluster } from "d3-hierarchy";
import * as Global from "./global";
import {
  smooth_pairwise_line_generator,
  statisticalDivision,
  scaleClustersBars, lowerIntersect, upperIntersect,
  areaMargin, placeImages, generateSortedEdges,
  roundSvgPath, line_interpolate,
} from "./layout_utils"

// MaHua: change line width
const scale_wide_line = 1.3;
const line_type = 0; // 0 line 1 mahua
const line_height_ratio = 1 // set line height ratio
const selected_action_height_ratio = 1 // set line height ratio

const lineGenerator = (points, width) => {
  const single_line = (input_points, prefix = "M") => {
    let points = input_points.map((d) => [d.x, d.y]);
    let s = `${prefix}${points[0].x},${points[0].y}`;
    s = points.reduce(
      (acc, point, i, a) =>
        i === 0
          ? `${prefix} ${point[0]},${point[1]}`
          : // : `${acc} ${bezierCommand(point, i, a)}`
            `${acc} ${smooth_pairwise_line_generator(point, i, a)}`,
      ""
    );
    return s;
  };
  if (width === undefined) {
    return single_line(points);
  } else {
    let top_points = points.map((d, i) => {
      return { x: d.x, y: d.y - width / 2 };
      // return {x:d.x, y: d.y};
    });
    let bottom_points = points.map((d, i) => {
      return { x: d.x, y: d.y + width / 2 };
    });
    top_points = top_points.reverse();
    bottom_points = bottom_points
    let top_s = single_line(top_points);
    let bottom_s = single_line(bottom_points, "L");
    // bottom_s[0] = "L";
    // return top_s ;
    // return bottom_s;
    return top_s + bottom_s + "Z";
  }
};

const get_all_descendants = function (node) {
  let to_visit = [node];
  let res = [];
  while (to_visit.length > 0) {
    let v = to_visit.pop();
    res.push(v);
    if (v.all_children && v.all_children.length > 0) {
      to_visit = v.all_children.concat(to_visit);
    }
  }
  return res;
};

function create_action(action_node, card, type, { x, y }) {
  let action = {};
  // action.action_id = card.depth + '_' + (action_node.id || (action_node.actions && action_node.actions[0]))
  action.action_id = action_node.id || (action_node.actions && action_node.actions[0])
  action.total_id = action_node.cur_index;
  action.selected = 0;
  action.width = card.width;
  action.height = card.line_height;
  action.band_height = card.y_layout_height;
  action.x = x
  action.y = y

  action.type = type;
  action.y_layout = action_node.y_layout;
  action.data = action_node.data;
  action.rep_frames = action_node.rep_frames;
  action.rep_action = action_node.rep_action;
  action.vid = action_node.vid;
  action.parent = card;
  action.width = type == 'leaf' ? 2 : 4 * scale_wide_line;
  if (action.data.children) { // TODO: replace it with a flag
    action.single_frame = action.rep_action.data.single_frame
    action.vid = action.rep_action.vid
    action.bound = action.rep_action.data.bound
  } else {
    action.single_frame = action.data.single_frame
    action.bound = action.data.bound
  }
  action.initial_bound = Global.deepCopy(action.bound)
  return action
}

function create_frame(d, { index, action, frame_index, y_layout, e_width }) {
  let frame = {}
  frame.fid = index;
  frame.id = `${action.total_id}-${index}`
  frame.mismatch = action.data.mismatch_scores && action.data.mismatch_scores[index] || 0
  frame.x = (index + 0.5) * e_width;
  frame.y = y_layout[action.total_id][index];
  frame.width = 10;
  frame.height = 10;
  frame.action_id = action.action_id
  frame.action_idx = action.action_idx;
  frame.video_left_bound = action.bound[0]
  frame.frame_idx = frame_index;
  frame.cluster_idx = action.cluster_idx;
  frame.is_action = d
  frame.is_dragging = 0
  frame.is_hidden = 0
  frame.is_rep = 0
  return frame
}

function create_frame_band({ x, y1, y2, cluster, band_width, frames, global_offset, band_idx }) {
  let frame_band = {
    offset: global_offset,
    frames,
    mismatch: frames.map(d => d.mismatch).reduce((a, b) => a + b),
    cluster_id: cluster.cluster_id,
    cluster_idx: cluster.cluster_idx,
    band_id: `${cluster.cluster_id}-${band_idx}`,
    fid: frames[0].fid,
    select_image_index: -1,
    x: x,
    y: y1,
    height: Math.max(10, y2 - y1),
    image_offset: { x: 0, y: 0 },
    width: band_width,
    show_image: false,
    show_band: false,
  }
  return frame_band
}

function generateClusterHull(lines, margin, hull_bended_edge) {
  let points = []
  for (let i = 0; i < lines.length; ++i) {
    let curr, last = null, last_my = margin
    for (let j = 0; j < lines[i].points.length; ++j) {
      curr = lines[i].points[j];
      let my = margin
      //if (curr.is_action > -1) {
        my *= 2
      //}

      points.push([curr.x, curr.y]);
      points.push([curr.x - margin * 4, curr.y - my]);
      points.push([curr.x - margin * 4, curr.y + my]);
      points.push([curr.x + margin * 4, curr.y - my]);
      points.push([curr.x + margin * 4, curr.y + my]);
      if (last) {
        let x1 = Math.min(curr.x, last.x)
        let y1 = Math.min(curr.y, last.y)
        let x2 = Math.max(curr.x, last.x)
        let y2 = Math.max(curr.y, last.y)
        let dx = x2 - x1
        let dy = y2 - y1
        let m = Math.ceil(dx / 50) + 1
        for (let k = 1; k < m; ++k) {
          points.push([
            x1 + dx * k / m - my,
            y1 + dy * k / m - my,
          ]);
          points.push([
            x1 + dx * k / m + my,
            y1 + dy * k / m - my,
          ]);
          points.push([
            x1 + dx * k / m - my,
            y1 + dy * k / m + my,
          ]);
          points.push([
            x1 + dx * k / m + my,
            y1 + dy * k / m + my,
          ]);
        }
      }
      last = curr;
    }
  }
  points = hull(points, hull_bended_edge)
  points.push(points[0])
  return points
}


const AlignmentLayout = function (parent) {
  let that = this;
  that.parent = parent;

  that.update_info_from_parent = function () {
    // states
    that.selected_class = that.parent.selected_class;
    that.use_treecut = that.parent.use_treecut;
    that.hierarchy = that.parent.hierarchy;
    that.selected_ids = that.parent.selected_ids;
    that.expand_node_ids = that.parent.expand_node_ids;
    that.data_id_mapping = that.parent.data_id_mapping;
    that.selected_cluster = that.parent.selected_cluster;
    that.anchor_alignments = that.parent.anchor_alignments;
    that.in_length_compare = that.parent.in_length_compare;

    // layout parameters
    that.matrix_height = that.parent.matrix_height;
    that.matrix_width = that.parent.matrix_width;
    that.tree_width = that.parent.matrix_width;
    that.matrix_width = that.parent.matrix_width;
    that.tree_width = that.parent.tree_width;
    that.alignment_width = that.parent.alignment_width - 20;
    that.layout_height = that.parent.layout_height;
    that.layout_width = that.parent.layout_width;
    that.frame_width = that.parent.frame_width;
    that.frame_height = that.parent.frame_height;
    that.large_frame_width = that.parent.frame_width * 2
    that.large_frame_height = that.parent.frame_height * 2
    that.ratio = that.parent.ratio;
    that.space = that.parent.space;

    that.margin = 30;
    that.image_margin = 25;
    that.snapshot_height = 100;
    that.single_layout_margin = 200;
    that.hull_bended_edge = 120
    that.hull_margin = 10;
    that.grid_size = 4
    that.base_margin = 10
    that.single_area_height = 65
    that.single_margin_bottom = that.margin * 2 + that.large_frame_height + that.single_area_height

    that.band_width = 24
    that.single_extend_offset = { x : -20, y: 20 }
    that.normal_extend_offset = { x : -10, y: 10 }
    that.extend_top_k = 4

    // animation
    that.create_ani = that.parent.create_ani;
    that.update_ani = that.parent.update_ani;
    that.remove_ani = that.parent.remove_ani;
  };
  that.update_info_from_parent();

  that.update_layout = function (data) {
    console.log("layout_alignment");
    that.update_info_from_parent();

    // if (that.selected_cluster.initial) return [];
    console.log("layout",data)
    if(data.children.length > 6) that.extend_top_k = 3;
    else that.extend_top_k = 4;

    that.data = data;
    that._layout();
    return that.cards;
  };

  that.update_lines = function() {
    that.cards.forEach(card => {
      card.actions.forEach((action) => {
        that._generate_center_line(action);
      });
    });
  }

  that._layout = function () {
    if (that.data.length === 0) return [];

    that.cards = [that.data];
    // console.log(that.data)
    that.cards.forEach((card, card_idx) => {
      card.type = !card.all_children ? "leaf" : "cluster"; // TODO: replace it with a is_leaf flag
      card.x = 0;
      card.y = 0;
      card.width = that.layout_width;
      card.single_area_height = that.single_area_height
      card.height = that.layout_height - that.margin;
      card.vis_height = that.layout_height - that.margin - 30;
      card.line_height = that.layout_height - that.margin * 2 - 30;
      card.y_layout_height = 1;
      let y_layout_height = Math.max(
        ...[].concat(...card.y_layout)
      );
      if (y_layout_height > 0) card.y_layout_height = y_layout_height;
    });

    // console.log('that.data', that.data)
    if (that.data.is_anchor_alignment) {
      that._single_layout();
    } else {
      that._normal_layout();
    }
  };

  that.primary_layout = function(normal_layout){
    console.log("primary_layout");
    that.layout_offset = normal_layout ? 0 : 50    
    that.cards.forEach((card) => {
      card.normal_layout = normal_layout
      if (!normal_layout) {
        card.line_height -= that.single_margin_bottom
      }
      card.actions = [];
      card.clusters = [];
      card.current_action = null;
      card.rep_frames = []
      let e_width =
        (that.ratio[1] * that.layout_width - 2 * that.margin) / card.y_layout[0].length;
      let y_layout = card.y_layout;
      let cur_index = 0;
      card.all_children.forEach((cluster) => {
        if (!cluster.all_children) {
          cluster.cur_index = cur_index++
        } else {
          cluster.all_children.forEach((action_node) => {
            action_node.cur_index = cur_index++
          })
        }
      })

      card.all_children.forEach((cluster, cluster_index) => {
        if (cluster.all_children) return
        let action = create_action(cluster, card, 'leaf', {
          x: that.ratio[0] * that.layout_width + that.margin + that.layout_offset,
          y: 10,
        })
        // action.action_id = `${normal_layout ? 'N' : 'S'}-${action.action_id}`
        // action.action_id = `${normal_layout ? 'N' : 'S'}${action.action_id}`
        if (!normal_layout && cluster_index == 0) {
          action.action_id = `${that.parent.last_action.action_id}`
        } else {
          action.action_id = `${normal_layout ? 'N' : 'S'}${action.action_id}`
        }
        if (!normal_layout) {
          if (cluster_index == 0) {
            action.width *= 3.5
          } else {
            action.width *= 2.5
          }
        }
        action.cluster_idx = cluster_index;
        action.action_idx = 0;

        let align_index = Global.deepCopy(card.aligns[cluster_index]);
        let visible_index = [];
        align_index.forEach((d, index) => {
          if(d > -1) visible_index.push(index);
        })
        let left_vis_bound = visible_index[0] - 1,
          right_vis_bound = visible_index[visible_index.length-1] + 1;

        action.frames = [];
        let frame_index = 0;
        let rep_action_index = 0;
        align_index.forEach((d, index) => {
          let frame = create_frame(d, { index, action, frame_index, y_layout, e_width })
          frame.is_single_frame = d > 0 && action.single_frame == rep_action_index + action.bound[0]
          frame.show_band = normal_layout && (index > left_vis_bound && index < right_vis_bound)? 1: 0;
          action.frames.push(frame);
          if (frame.is_action > -1) {
            frame_index++;
            if (frame.is_action > 0) {
              rep_action_index++;
            }
          }
        });
        if(!cluster.rep_cols) cluster.rep_cols = []
        if(!cluster.rep_rows) cluster.rep_rows = []
        card.clusters.push({
          cluster_id: cluster.id || (cluster.actions && cluster.actions[0]),
          cluster_idx: cluster_index,
          actions: [action],
          bars: normal_layout ? statisticalDivision(cluster.action_lengths) : [],
          cbars: normal_layout && that.in_length_compare? statisticalDivision(cluster.compare_lengths) : [],
          data: cluster,
          rep_cols: cluster.rep_cols.map(d => visible_index[d]),
          rep_rows: cluster.rep_rows,
        });
        normal_layout && scaleClustersBars(card.clusters, that.in_length_compare)
        card.actions.push(action);
        return;
      });
    })
  };

  that._single_layout = function() {
    that.primary_layout(false);
    console.log('_single_layout')
    // console.log('compared item', that.parent.last_action)
    that.cards.forEach((card) => {
      card.actions.forEach((action) => {
        // set line height ratio
        action.frames.forEach((frame) => {
          frame.y = frame.y * card.line_height * line_height_ratio;
        });
        action._frames = action.frames;
        action.is_anchor = false;

        that._generate_center_line(action);
      });
      card.actions[0].is_anchor = true;
      // console.log('card.actions', card.actions)
  
      let cluster_rep_frames = []
      card.clusters.forEach((cluster, cluster_index) => {
        cluster.x = cluster.actions[0].x;
        cluster.y = cluster.actions[0].y;
        let points = [];
        cluster.points = points
        cluster.frames = []

        let rep_frames = []
        cluster.actions.forEach((action) => {
          const rep_frame_set = new Set(action.rep_frames)
          action.frames
            .filter(d => d.is_action > 0)
            .forEach((d, index) => {
                const vid = action.vid
                const fid = action.bound[0] + index
                if (d.is_action > 0 && rep_frame_set.has(`${vid}-${fid}`)) {
                    d.is_rep = 1
                }
                d.id = `${vid}-${fid}`
            })
            action.frames
              .forEach((d, index) => {
                d.is_selected = 0
              })
          rep_frames = rep_frames.concat(action.frames.filter(d => d.is_action > 0))
        });

        rep_frames = rep_frames.sort((a, b) => a.fid - b.fid)
        const merged_rep_frames = []
        let last_image_x = -1e10
        for (let i = 0; i < rep_frames.length; ++i) {
          let left = i, right = i + 1
          while (right < rep_frames.length && rep_frames[left].fid >= rep_frames[right].fid) {
            ++right
          }
          let frames = rep_frames.slice(left, right)
          const x = Math.max(...frames.map(d => d.x))
          let y1 = Math.min(...frames.map(d => d.y))
          let y2 = Math.max(...frames.map(d => d.y))
          
          const frame_band = 
          create_frame_band({ x, y1, y2, cluster, frames,
            band_width: that.band_width,
            global_offset: {
              x: that.ratio[0] * that.layout_width + that.margin + that.layout_offset,
              y: 10,
            },
            band_idx: merged_rep_frames.length
          })
          frame_band.image_offset = that.single_extend_offset
          if (cluster_index == 0 && frames.length > 0) {
            if (x - last_image_x < that.frame_width) {
              frame_band.show_image = 0
              frames[0].is_selected = 0
              frame_band.select_image_index = -1
            } else {
              frame_band.select_image_index = 0
              frame_band.show_image = 1
              frames[0].is_selected = 1
              last_image_x = x
            }
          }
          frames.forEach((d, index) => {
            d.index = index
          })
          merged_rep_frames.push(frame_band)
          i = right - 1
        }
        // console.log('merged_rep_frames', merged_rep_frames)
        card.rep_frames = card.rep_frames.concat(merged_rep_frames)
        cluster_rep_frames.push({
          index: cluster_index,
          count: rep_frames.length, // filter(d => rep_frame_set.has(d.id)).length,
          merged_rep_frames: merged_rep_frames,
        })
      });

      card.rep_frames.forEach((frame, id) => {
        frame.index = id
      })
      const margin = that.single_layout_margin

      card.rep_frames.forEach(rep => {
        rep.y = (rep.y + margin) / (card.line_height + margin * 2) * card.line_height
        rep.height = rep.height / (card.line_height + margin * 2) * card.line_height
      })

      card.actions.forEach((action) => {
        action.frames.forEach((frame) => {
          frame.y = (frame.y + margin) / (card.line_height + margin * 2) * card.line_height
        });
      });
      card.clusters.forEach(cluster => {
        cluster.convex_hull = ''
      })

      card.clusters.forEach(cluster => {
        cluster.frames = []
        cluster.actions.forEach(action => {
          cluster.frames = cluster.frames.concat(action.frames)
        })
        // console.log('cluster.frames', cluster.frames)
      })
      const all_frames = [].concat(...card.clusters.map(d => d.frames)).filter(d => d.is_action > 0)
      const x_range = [
        Math.min(...all_frames.map(d => d.x)),
        Math.max(...all_frames.map(d => d.x))
      ]
      const comparsion_frames = all_frames.filter(d => d.cluster_idx > 0)
      const frame_y = comparsion_frames.map(d => d.y)
        .concat([margin, card.line_height - margin])
        .sort((a, b) => a - b)
      let max_margin = 0, k = 0
      for (let i = 0; i < frame_y.length - 1; ++i) {
        if (frame_y[i + 1] - frame_y[i] > max_margin) {
          max_margin = frame_y[i + 1] - frame_y[i]
          k = i
        }
      }
      const padding = 25
      card.must_link_area = {
        x1: x_range[0] - padding,
        x2: x_range[1] + padding,
        y1: frame_y[0] - padding,
        y2: frame_y[k] + padding,
      }

      card.must_not_link_area = {
        x1: x_range[0] - padding,
        x2: x_range[1] + padding,
        y1: frame_y[k + 1] - padding,
        y2: frame_y[frame_y.length - 1] + padding,
      }
    });
    that.update_lines();
  };

  that._normal_layout = function () {
    that.primary_layout(true);
    console.log('_normal_layout')
    that.cards.forEach((card) => {
      console.assert(card.type === "cluster", "Card shouldn't be leaf node.");

      let e_width =
        (that.ratio[1] * that.layout_width - 2 * that.margin) / card.y_layout[0].length;
      let y_layout = card.y_layout;
      card.all_children.forEach((cluster, cluster_index) => {
        if (!cluster.all_children) return
        let actions = []
        cluster.all_children.forEach((action_node, action_index) => {
          let action = create_action(action_node, card, 
            !action_node.all_children ? "leaf" : "cluster", 
            {
              x: that.ratio[0] * that.layout_width + that.margin,
              y: 10,
          })
          // action.action_id = `N-${action.action_id}`
          action.action_id = `N${action.action_id}`
          action.action_idx = action_index;
          action.cluster_idx = cluster_index;
          let align_index = Global.deepCopy(card.aligns[cluster_index]);
          let w = 0,
            t = 0;
          align_index = align_index.map((d) => {
            if (d != -1 && cluster.aligns[action_index][w++] != -1) {
              if (action_node.rep_action_position[t++] != -1) return 1;
              else return 0;
            }
            return -1;
          });
          let visible_index = [];
          align_index.forEach((d, index) => {
            if(d > -1)
              visible_index.push(index);
          })
          let left_vis_bound = visible_index[0] - 1,
            right_vis_bound = visible_index[visible_index.length-1] + 1;
          action.frames = [];
          let frame_index = 0;
          let rep_action_index = 0;
          align_index.forEach((d, index) => {
            let frame = create_frame(d, { index, action, frame_index, y_layout, e_width })
            frame.is_single_frame = d > 0 && action.single_frame === rep_action_index + action.bound[0]
            frame.show_band = (index >left_vis_bound && index < right_vis_bound)? 1: 0;
            action.frames.push(frame);
            if (frame.is_action > -1) {
              frame_index++;
              if (frame.is_action > 0) {
                rep_action_index++;
              }
            }
          });
          actions.push(action);
        });
        let align_index = Global.deepCopy(card.aligns[cluster_index]);
        let visible_index = [];
        align_index.forEach((d, index) => {
          if(d > -1) visible_index.push(index);
        })
        card.clusters.push({
          cluster_id: cluster.id,
          cluster_idx: cluster_index,
          actions,
          bars: statisticalDivision(cluster.action_lengths),
          cbars: that.in_length_compare ? statisticalDivision(cluster.compare_lengths): [],
          data: cluster,
          rep_cols: cluster.rep_cols.map(d => visible_index[d]),
          rep_rows: cluster.rep_rows,
        });
        scaleClustersBars(card.clusters, that.in_length_compare)
        card.actions = card.actions.concat(actions);
      });

      // resort clusters sequence
      card.clusters.sort((a, b) => a.cluster_idx - b.cluster_idx)
      card.actions.sort((a, b) => a.cluster_idx == b.cluster_idx ?
        (a.action_idx - b.action_idx) : (a.cluster_idx - b.cluster_idx))
      console.log('card.clusters', card.clusters)
      console.log('card.actions', card.actions)
      // console.log(card.clusters)

      card.actions.forEach((action) => {
        action.frames.forEach((frame) => {
          frame.y = (frame.y / card.y_layout_height) * card.line_height;
        });
        action._frames = action.frames;

        that._generate_center_line(action);
      });
  
      let cluster_rep_frames = []
      card.clusters.forEach((cluster, cluster_index) => {
        cluster.x = cluster.actions[0].x;
        cluster.y = cluster.actions[0].y;
        let points = generateClusterHull(cluster.actions, that.hull_margin, that.hull_bended_edge)
        cluster.points = points
        cluster.lines = generateSortedEdges(points)
        cluster.frames = []

        let rep_frames = []
        cluster.actions.forEach((action) => {
          const rep_frame_set = new Set(action.rep_frames)
          action.frames
            .filter(d => d.is_action > 0)
            .forEach((d, index) => {
                const vid = action.vid
                const fid = action.bound[0] + index
                if (d.is_action > 0 && rep_frame_set.has(`${vid}-${fid}`)) {
                    d.is_rep = 1
                }
                d.id = `${vid}-${fid}`
            })
            action.frames
              .forEach((d, index) => {
                d.is_selected = 0
              })
          rep_frames = rep_frames.concat(action.frames.filter(d => d.is_action > 0))
        });
        // console.log("rep_frames", rep_frames);

        rep_frames = rep_frames.sort((a, b) => a.fid - b.fid)
        const min_fid = rep_frames.filter(d => d.is_action > -1)[0].fid
        const merged_rep_frames = []
        for (let i = 0; i < rep_frames.length; ++i) {
          let left = i, right = i + 1
          while (right < rep_frames.length && rep_frames[left].fid >= rep_frames[right].fid) {
            ++right
          }
          let frames = rep_frames.slice(left, right)
          const x = Math.max(...frames.map(d => d.x))
          let y1 = lowerIntersect([x], cluster.lines)[0]
          let y2 = upperIntersect([x], cluster.lines)[0]
          
          const frame_band = 
            create_frame_band({ x, y1, y2, cluster, frames,
              band_width: that.band_width,
              global_offset: {
                x: that.ratio[0] * that.layout_width + that.margin,
                y: 10,
              },
              band_idx: merged_rep_frames.length
            })
          // frame_band.show_band = frames.filter(d => d.is_rep).length > 0
          let rep_idx = cluster.rep_cols.indexOf(frame_band.fid);
          frame_band.show_band = rep_idx !== -1

          if (frame_band.show_band) {
            let rep_row = cluster.rep_rows[rep_idx]
            let frame_init_idx = 0;
            // console.log("sfsfa", frames, rep_row)
            frames.forEach((d, index) => {
              if(d.action_idx === rep_row){
                frame_init_idx = index
              }
            })
            frames[frame_init_idx].is_selected = 1;
            frame_band.select_image_index = frame_init_idx;
          }
          frames.forEach((d, index) => {
            d.index = index
          })
          merged_rep_frames.push(frame_band)
          i = right - 1
        }

        // console.log(merged_rep_frames.map(d => d.x))
        if (merged_rep_frames.filter(d => d.show_band).length > 12) {
          merged_rep_frames.sort((a, b) => b.frames.length - a.frames.length)
            .slice(12).forEach(d => {
              d.show_band = false
              if (d.frames.length == 0) return
              d.frames[0].is_selected = 0
              d.select_image_index = -1
            })
        }
        // console.log('band', merged_rep_frames.filter(d => d.show_band))

        merged_rep_frames.sort((a, b) => a.fid - b.fid)
        card.rep_frames = card.rep_frames.concat(merged_rep_frames)
        cluster_rep_frames.push({
          index: cluster_index,
          count: rep_frames.length, // filter(d => rep_frame_set.has(d.id)).length,
          merged_rep_frames: merged_rep_frames,
        })
      });
      // console.log('cluster_rep_frames',cluster_rep_frames)

      if(!card.cluster_rep_frame_indexes){
        card.cluster_rep_frame_indexes = cluster_rep_frames
          .sort((a, b) => b.count - a.count)
          .slice(0, that.extend_top_k)
          .map(d => d.index);
      }
      const cluster_rep_frame_indexes = card.cluster_rep_frame_indexes;
      const cluster_rep_frame_indexes_set = new Set(cluster_rep_frame_indexes)
    
      card.rep_frames.forEach((rep_frame, id) => {
        rep_frame.index = id
        if (rep_frame.show_band) {
          rep_frame.show_band = cluster_rep_frame_indexes_set.has(rep_frame.cluster_idx)
        }
        if (!rep_frame.show_band) {
          rep_frame.frames.forEach(d => d.is_selected = 0)
        }
      })
      
      for (let i = 0; i < card.rep_frames.length; ++i) {
        card.rep_frames[i].show_image = card.rep_frames[i].show_band
      }
      

      // adjust margin between clusters
      let max_y = 0, extend_y = 0
      card.clusters.forEach((cluster, cluster_index) => {
        cluster.extend_y = that.base_margin
        cluster.min_margin = 0
        for (let i = 0; i < cluster.points.length; ++i) {
          if (cluster.points[i][1] > max_y) {
            max_y = cluster.points[i][1]
          }
        }
        if (!cluster_rep_frame_indexes_set.has(cluster_index)) {
          return
        }
        let frames = card.rep_frames
          .filter(frame => frame.cluster_idx == cluster_index)
          .filter(frame => frame.show_band)
        let margins
        if (cluster_index == 0) {
          margins = lowerIntersect(frames.map(d => d.x), cluster.lines)
        } else {
          margins = areaMargin(
            [].concat(...frames.map(d => [d.x - that.frame_width / 2, d.x + that.frame_width / 2])),
            card.clusters[cluster_index - 1].lines,
            cluster.lines
          )
        }
        let min_margin = Math.max(0, Math.min(...margins))
        cluster.min_margin = min_margin  
        const margin = that.frame_height + that.image_margin
        if (min_margin < margin) {
          cluster.extend_y = margin - min_margin
        }

        extend_y += cluster.extend_y
      })
      
      if (cluster_rep_frame_indexes_set.has(0))
        card.clusters[0].extend_y = that.frame_height + that.image_margin * 0.5
      for (let i = 1; i < card.clusters.length; ++i) {
        card.clusters[i].extend_y += card.clusters[i - 1].extend_y
        card.clusters[i].min_margin += card.clusters[i - 1].min_margin
      }

      const sum_margin = card.clusters[card.clusters.length - 1].min_margin
      // console.log('max_y', max_y, 'sum margin', sum_margin, 'extend_y', extend_y)
      const ratio = 1.0 / (max_y) * (max_y - extend_y)
  
      // console.log('card.clusters', card.clusters)
      card.clusters.forEach(cluster => {
        const points = cluster.points
        for (let i = 0; i < points.length; ++i) {
          points[i] = [
            points[i][0],
            (points[i][1] - cluster.min_margin) * ratio + cluster.extend_y + cluster.min_margin
          ]
        }
      })

      card.rep_frames.forEach(rep => {
        let cluster = card.clusters[rep.cluster_idx]
        rep.y = (rep.y - cluster.min_margin) * ratio + cluster.extend_y + cluster.min_margin
        rep.height = rep.height / (max_y) * (max_y - extend_y)
      })

      card.actions.forEach((action) => {
        let cluster = card.clusters[action.cluster_idx]
        action.frames.forEach((frame) => {
          frame.y = (frame.y - cluster.min_margin) * ratio + cluster.extend_y + cluster.min_margin
        });
      });

      card.clusters.forEach((cluster) => {
        cluster.lines = generateSortedEdges(cluster.points)
      })

      card.clusters.forEach(cluster => {
        cluster.frames = []
        cluster.actions.forEach(action => {
          cluster.frames = cluster.frames.concat(action.frames)
        })
      })

      card.clusters.forEach((cluster, cluster_index) => {
        let frames = card.rep_frames
          .filter(frame => frame.cluster_idx == cluster_index)
          .filter(frame => frame.show_band)
        if (frames.length == 0) return
        let prev_lines = cluster_index == 0 ? null : card.clusters[cluster_index - 1].lines

        let offset = null
        while (1) {
          offset = placeImages(frames, prev_lines, cluster.lines,
          {
            grid_size: 4,
            area_width: card.width - that.frame_width,
            image_width: that.frame_width,
            image_height: that.frame_height,
            padding: 4
          })
          if (offset) break
          const delta = 20
          for (let point of cluster.points) {
            point[1] += delta
          }
          cluster.lines = generateSortedEdges(cluster.points)
          for (let frame of cluster.frames) {
            frame.y += delta
          }
          for (let item of cluster_rep_frames) {
            if (item.index == cluster_index) {
              for (let band of item.merged_rep_frames) {
                band.y += delta
              }
            }
          }
        }
        frames.forEach((d, index) => {
          d.image_offset = offset[index]
        })
      })

      card.clusters.forEach(cluster => {
        let points = cluster.points
        let convex_hull = `M${points[0][0]},${points[0][1]} `;
        for (let curr of points.slice(1)) {
          convex_hull += `L${curr[0]},${curr[1]} `;
        }

        convex_hull = convex_hull + `L${points[0][0]},${points[0][1]} ` + "Z";
        cluster.convex_hull = roundSvgPath(convex_hull, 100, "circle");
      })

      // calculate bar chart x and y
      let r = 0.5;
      card.clusters.forEach(d => {
        let xmax = 0;
        let ymin = 2000, ymax = 0;
        d.actions.forEach(e => {
          e.frames.forEach(f => {
            if(f.x > xmax) xmax = f.x;
          })
        })
        d.actions.forEach(e => {
          e.frames.filter(d => d.x > r * xmax).forEach(f => {
            if(f.y < ymin) ymin = f.y;
            if(f.y > ymax) ymax = f.y;
          })
        })
        d.bar_chart_x = xmax;
        d.bar_chart_y = (ymin + ymax) / 2;
        if (d.bar_chart_y < 55 && that.in_length_compare)
          d.bar_chart_y = 55
        if (d.bar_chart_y > that.layout_height - 80) 
          d.bar_chart_y = that.layout_height - 80;
      })

      let new_actions = card.clusters.map(cluster => {
        let frames = cluster.frames.filter(d => !d.is_hidden).sort((a, b) => a.x - b.x)
        let filtered = []
        for (let i = 0; i < frames.length; ++i) {
          let j = i
          while (j + 1 < frames.length && frames[i].x == frames[j + 1].x) ++j
          filtered.push(frames[Math.floor((i + j) / 2)])
          i = j
        }
        frames = filtered
        return {
          action_id: `N${cluster.cluster_id}`,
          type: 'fake',
          frames,
          width: cluster.actions[0].width,
          height: cluster.actions[0].height,
          band_height: [],
          x: cluster.actions[0].x,
          y: cluster.actions[0].y,
        }
      })
      // card.actions = card.actions.concat(new_actions)
    });
    that.update_lines();
  };

  that._generate_center_line = function (action) {
    let points = action.frames.map((d) => {
      let res = {
        x: d.x, y: d.y,
        id: d.fid,
        is_action: d.is_hidden ? 0 : d.is_action,
        is_dragging: d.is_dragging
      };
      return res;
    });
    // console.log('points', points)
    let widths = points.map((d) => 0);
    let count = 0;
    let min_nonzero = 1000;
    let max_nonzero = -1;
    for (let i = 0; i < widths.length; i++) {
      if (points[i].is_action > 0) {
        widths[i] = action.band_height[count++];
      }
      if (points[i].is_action >= 0) {
        if (i < min_nonzero) min_nonzero = i;
        if (i > max_nonzero) max_nonzero = i;
      }
    }
    // if (min_nonzero < widths.length) min_nonzero--;
    // if (max_nonzero + 1 < widths.length) max_nonzero++;
    widths = line_interpolate(widths);
    let old_points = points
    points = points.slice(min_nonzero, max_nonzero + 1);
    widths = widths.slice(min_nonzero, max_nonzero + 1);
    action.widths = widths;
    let no_dragging = true
    for (let i = 0; i < points.length; ++i) {
      if (points[i].is_action == -1) continue
      if (points[i].is_dragging) {
        no_dragging = false
        points = points.filter((_, j) => j == i || (j < i && points[j].x < points[i].x) || (i < j && points[i].x < points[j].x))
        break
      }
    }

    
    if (no_dragging) {
      const min_margin = 10
      for (let i = 0, x = 0; i < points.length; ++i) {
        points[i].remove = points[i].x - min_margin <= x
        if (!points[i].remove && points[i].is_action > 0) {
          x = Math.max(points[i].x, x)
        }
      }
      for (let i = points.length - 1, x = 1e10; i >= 0; --i) {
        points[i].remove = points[i].remove || points[i].x + min_margin >= x
        if (!points[i].remove && points[i].is_action > 0) {
          x = Math.min(points[i].x, x)
        }
      }
      points = points.filter(d => !d.remove)
      // console.log('points', points)
    }
    
    // console.log('points', points)
    action.center_line = lineGenerator(points)
    action.center_line_z = lineGenerator(points, 5)
    action.start_line = ''
    action.end_line = ''
    let tail_len = 20;
    // if (min_nonzero > 0) {
    //   let p0 = { x: old_points[min_nonzero - 1].x, y: old_points[min_nonzero - 1].y }
    //   let p1 = old_points[min_nonzero]
    //   p0 = { x: p1.x - tail_len, y: p1.y - (p1.y - p0.y) / (p1.x - p0.x) * tail_len }
    //   action.start_line = lineGenerator([p0, p1])
    // }
    let p1 = {x: old_points[min_nonzero].x, y: old_points[min_nonzero].y}
    let p0 = {x: p1.x - tail_len, y: p1.y};
    action.start_line = lineGenerator([p0, p1]);
    p0 = {x: old_points[max_nonzero].x, y: old_points[max_nonzero].y};
    p1 = {x: p0.x + tail_len, y: p0.y};
    action.end_line = lineGenerator([p1, p0]);
    action.points = points;
    
    if(line_type === 0)
      that._generate_copy_line(action, points)
    else
      that._generate_bundled_line(action, points);
  };

  that._generate_copy_line = function(action, points){
    let delta = 1.3 * scale_wide_line;
    let pts1 = [], pts2 = [];
    let dy1 = delta, dy2 = -delta;
    points.forEach((value, index) => {
      pts1.push({
        x: value.x,
        y: value.y + dy1
      })
      pts2.push({
        x: value.x,
        y: value.y + dy2
      })
    })

    action.center_line_1 = lineGenerator(pts1);
    action.center_line_2 = lineGenerator(pts2);
  }

  that._generate_bundled_line = function(action, points){
    // MaHua: change mahua width
    let delta = 1.3 * scale_wide_line;
    
    let pts1 = [], pts2 = [];
    let dy1 = delta, dy2 = -delta;
    points.forEach((value, index) => {
      if(index === 0){
        pts1.push({
          x: value.x,
          y: value.y + dy1
        })
        pts2.push({
          x: value.x,
          y: value.y + dy2
        })
      }
      else{
        pts1.push({
          x: value.x,
          y: value.y + dy1
        })
        pts2.push({
          x: value.x,
          y: value.y + dy2
        })
      }
      dy1 = dy1 * -1;
      dy2 = dy2 * -1;
    })

    action.center_line_1 = lineGenerator(pts1);
    action.center_line_2 = lineGenerator(pts2);
  }
};

export { AlignmentLayout };
