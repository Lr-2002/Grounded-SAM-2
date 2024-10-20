# How to use the metric 

## prepare data 
all the data should be prepared use the grounding sam2 to generate all the npz data 
the data should be contained in  a dir with name is frame%06d.npz to save 
│   ├── frame000000.npz
│   ├── frame000001.npz
│   ├── frame000002.npz
│   ├── frame000003.npz
│   ├── frame000004.npz
│   ├── frame000005.npz
│   ├── frame000006.npz
│   ├── frame000007.npz
│   ├── frame000008.npz
│   ├── frame000009.npz
│   ├── frame000010.npz
│   ├── frame000011.npz
│   ├── frame000012.npz
│   ├── frame000013.npz
│   ├── frame000014.npz
│   └── frame000015.npz

## run the metric 
fix the path of the upper folder 
and use the num_metrics file to calculate the file-level miss, hide and addon
