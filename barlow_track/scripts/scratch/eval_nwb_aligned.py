"""Run eval_accuracy.py with name-aligned NWB GT arrays, without editing it.

eval_accuracy._load_nwb_arrays takes each GT field with df.loc[:, (slice(None), field)],
whose neuron order follows the frame's columns. In the flavell NWB the id fields
(raw_neuron_ind_in_list, raw_segmentation_id) come out in a different neuron order than
x/y/z, so crops are labelled with another neuron's ids and ~9% of GT entries are never
cropped (see nwb_gt_coverage.py). This wrapper replaces _load_nwb_arrays with a version
that selects every field by neuron name, then calls eval_accuracy.main() with the same CLI.

Usage (wbfm env): same arguments as eval_accuracy.py, e.g.
  python eval_nwb_aligned.py --lab flavell --source nwb --nwb ... --mode trained --weights ...
"""
import barlow_track.scripts.eval_accuracy as ea


def _load_nwb_arrays_aligned(nwb_project):
    df_gt = ea._ensure_ind_level(nwb_project.final_tracks)
    neurons = list(df_gt.columns.get_level_values(0).unique())

    def field(name):
        return df_gt.loc[:, [(n, name) for n in neurons]].values.astype(float)

    return dict(df_gt=df_gt, neurons=neurons, gx=field('x'), gy=field('y'), gz=field('z'),
                gr=field('raw_neuron_ind_in_list'), gs=field('raw_segmentation_id'))


if __name__ == '__main__':
    ea._load_nwb_arrays = _load_nwb_arrays_aligned
    ea.main()
