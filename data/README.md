# Data format

Raw and processed data are excluded from version control.

## Processed graph file

Each subject is stored as a PyTorch `.pt` dictionary:

```python
{
    "x_derivative": FloatTensor[num_rois, target_time],
    "x_raw_zscore": FloatTensor[num_rois, target_time],
    "edge_list": [(0, 4), (2, 8, 17), ...],
    "edge_p_values": FloatTensor[num_edges],
    "label": 0 or 1,
    "dataset": "ABIDE_I",
    "group": "control" or "patient",
    "subject_id": "subject_identifier",
}
```

ROI indices in `edge_list` are zero-based. Node features follow the `|V| x T` convention.

## Manifest

The training scripts expect a CSV with at least these columns:

```text
status,dataset,subject_id,group,label,file,pt_path
ok,ABIDE_I,sub001,control,0,control/timeseries/sub001.txt,graphs/control/sub001.pt
```

`pt_path` may be absolute or relative to the manifest directory.

