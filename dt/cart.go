package dt

import (
	"bufio"
	"container/list"
	"fmt"
	"github.com/xlvector/hector/core"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
)

/*
CART is classification and regression tree, this class implement classification tree and use gini
to split features
*/
type CART struct {
	tree                Tree
	params              CARTParams
	continuous_features bool
	salt                int64
}

func DTGoLeft(sample *core.MapBasedSample, feature_split core.Feature) bool {
	value, ok := sample.Features[feature_split.Id]
	if ok && value >= feature_split.Value {
		return true
	} else {
		return false
	}
}

func DTGetElementFromQueue(queue *list.List, n int) []*TreeNode {
	ret := []*TreeNode{}
	for i := 0; i < n; i++ {
		node := queue.Front()
		if node == nil {
			break
		}
		ret = append(ret, (node.Value.(*TreeNode)))
		queue.Remove(node)
	}
	return ret
}

func (dt *CART) RandByFeatureId(fid int64) float64 {
	ret := fid*19857 + dt.salt
	r := math.Abs(float64(ret%1000) / 1000.0)
	return r
}

func (dt *CART) FindBestSplitOfContinusousFeature(samples []*core.MapBasedSample, node *TreeNode, feature_select_prob float64) {
	feature_weight_labels := make(map[int64]*core.FeatureLabelDistribution)
	total_dis := core.NewArrayVector()
	for i, k := range node.samples {
		if i > 10 && rand.Float64() > dt.params.SamplingRatio {
			continue
		}
		total_dis.AddValue(samples[k].Label, 1.0)
		for fid, fvalue := range samples[k].Features {
			if dt.RandByFeatureId(fid) > feature_select_prob {
				continue
			}
			_, ok := feature_weight_labels[fid]
			if !ok {
				feature_weight_labels[fid] = core.NewFeatureLabelDistribution()
			}
			feature_weight_labels[fid].AddWeightLabel(fvalue, samples[k].Label)
		}
	}

	min_gini := 1.0
	node.feature_split = core.Feature{Id: -1, Value: 0}
	for fid, distribution := range feature_weight_labels {
		sort.Sort(distribution)
		split, gini := distribution.BestSplitByGini(total_dis)
		if min_gini > gini {
			min_gini = gini
			node.feature_split.Id = fid
			node.feature_split.Value = split
		}
	}
	if min_gini > dt.params.GiniThreshold {
		node.feature_split.Id = -1
		node.feature_split.Value = 0.0
	}
}

func (dt *CART) FindBestSplitOfBinaryFeature(samples []*core.MapBasedSample, node *TreeNode, feature_select_prob float64) {
	feature_right_dis := make(map[int64]*core.ArrayVector)
	total_dis := core.NewArrayVector()
	for i, k := range node.samples {
		if i > 10 && rand.Float64() > dt.params.SamplingRatio {
			continue
		}
		total_dis.AddValue(samples[k].Label, 1.0)
		for fid, _ := range samples[k].Features {
			if dt.RandByFeatureId(fid) > feature_select_prob {
				continue
			}
			_, ok := feature_right_dis[fid]
			if !ok {
				feature_right_dis[fid] = core.NewArrayVector()
			}
			feature_right_dis[fid].AddValue(samples[k].Label, 1.0)
		}
	}

	min_gini := 1.0
	node.feature_split = core.Feature{Id: -1, Value: 0}
	for fid, right_dis := range feature_right_dis {
		left_dis := total_dis.Copy()
		left_dis.AddVector(right_dis, -1.0)
		gini := core.Gini(left_dis, right_dis)
		if min_gini > gini {
			min_gini = gini
			node.feature_split.Id = fid
			node.feature_split.Value = 1.0
		}
	}
	if min_gini > dt.params.GiniThreshold {
		node.feature_split.Id = -1
		node.feature_split.Value = 0.0
	}
}

func (dt *CART) AppendNodeToTree(samples []*core.MapBasedSample, node *TreeNode, queue *list.List, tree *Tree, feature_select_prob float64) {
	if node.depth >= dt.params.MaxDepth {
		return
	}

	if dt.continuous_features {
		dt.FindBestSplitOfContinusousFeature(samples, node, feature_select_prob)
	} else {
		dt.FindBestSplitOfBinaryFeature(samples, node, feature_select_prob)
	}
	if node.feature_split.Id < 0 {
		return
	}
	left_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: nil, sample_count: 0, samples: []int{}}
	right_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: nil, sample_count: 0, samples: []int{}}

	left_node.prediction = core.NewArrayVector()
	right_node.prediction = core.NewArrayVector()
	for _, k := range node.samples {
		if DTGoLeft(samples[k], node.feature_split) {
			left_node.samples = append(left_node.samples, k)
			left_node.prediction.AddValue(samples[k].Label, 1.0)
		} else {
			right_node.samples = append(right_node.samples, k)
			right_node.prediction.AddValue(samples[k].Label, 1.0)
		}
	}
	node.samples = nil

	if len(left_node.samples) > dt.params.MinLeafSize {
		left_node.sample_count = len(left_node.samples)
		left_node.prediction.Scale(1.0 / left_node.prediction.Sum())
		queue.PushBack(&left_node)
		node.left = len(tree.nodes)
		tree.AddTreeNode(&left_node)
	}

	if len(right_node.samples) > dt.params.MinLeafSize {
		right_node.sample_count = len(right_node.samples)
		right_node.prediction.Scale(1.0 / right_node.prediction.Sum())
		queue.PushBack(&right_node)
		node.right = len(tree.nodes)
		tree.AddTreeNode(&right_node)
	}
}

func (dt *CART) SingleTreeBuild(samples []*core.MapBasedSample, feature_select_prob float64, bootstrap bool) Tree {
	tree := Tree{}
	queue := list.New()
	root := TreeNode{depth: 0, left: -1, right: -1, prediction: core.NewArrayVector(), samples: []int{}}

	if !bootstrap {
		for i, sample := range samples {
			root.AddSample(i)
			root.prediction.AddValue(sample.Label, 1.0)
		}
	} else {
		for i := 0; i < len(samples); i++ {
			k := rand.Intn(len(samples))
			root.AddSample(k)
			root.prediction.AddValue(samples[k].Label, 1.0)
		}
	}
	root.sample_count = len(root.samples)
	root.prediction.Scale(1.0 / root.prediction.Sum())

	queue.PushBack(&root)
	tree.AddTreeNode(&root)
	for {
		nodes := DTGetElementFromQueue(queue, 10)
		if len(nodes) == 0 {
			break
		}

		for _, node := range nodes {
			dt.AppendNodeToTree(samples, node, queue, &tree, feature_select_prob)
		}
	}
	return tree
}

func PredictBySingleTree(tree *Tree, sample *core.MapBasedSample) (*TreeNode, string) {
	path := ""
	node := tree.GetNode(0)
	path += node.ToString()
	for {
		if DTGoLeft(sample, node.feature_split) {
			if node.left >= 0 && node.left < tree.Size() {
				node = tree.GetNode(node.left)
				path += "-" + node.ToString()
			} else {
				break
			}
		} else {
			if node.right >= 0 && node.right < tree.Size() {
				node = tree.GetNode(node.right)
				path += "+" + node.ToString()
			} else {
				break
			}
		}
	}
	return node, path
}

func (dt *CART) Train(dataset *core.DataSet) {
	samples := []*core.MapBasedSample{}
	feature_weights := make(map[int64]float64)
	for _, sample := range dataset.Samples {
		if !dt.continuous_features {
			for _, f := range sample.Features {
				_, ok := feature_weights[f.Id]
				if !ok {
					feature_weights[f.Id] = f.Value
				}
				if feature_weights[f.Id] != f.Value {
					dt.continuous_features = true
				}
			}
		}
		msample := sample.ToMapBasedSample()
		samples = append(samples, msample)
	}
	if dt.continuous_features {
		fmt.Println("Continuous DataSet")
	} else {
		fmt.Println("Binary DataSet")
	}
	dt.tree = dt.SingleTreeBuild(samples, 1.0, false)
}

func (dt *CART) Predict(sample *core.Sample) float64 {
	msample := sample.ToMapBasedSample()
	node, _ := PredictBySingleTree(&dt.tree, msample)
	return node.prediction.GetValue(1)
}

func (dt *CART) PredictMultiClass(sample *core.Sample) *core.ArrayVector {
	msample := sample.ToMapBasedSample()
	node, _ := PredictBySingleTree(&dt.tree, msample)
	return node.prediction
}

func (self *CART) SaveModel(path string) {
	ioutil.WriteFile(path, self.tree.ToString(), 0600)
}

func (self *CART) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()
	text := ""
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text += scanner.Text() + "\n"
	}
	self.tree.FromString(string(text))
}

type CARTParams struct {
	MaxDepth      int
	MinLeafSize   int
	GiniThreshold float64
	SamplingRatio float64
}

func (dt *CART) Init(params map[string]string) {
	dt.tree = Tree{}
	dt.continuous_features = false
	min_leaf_size, _ := strconv.ParseInt(params["min-leaf-size"], 10, 32)
	max_depth, _ := strconv.ParseInt(params["max-depth"], 10, 32)

	dt.params.MinLeafSize = int(min_leaf_size)
	dt.params.MaxDepth = int(max_depth)
	dt.params.GiniThreshold, _ = strconv.ParseFloat(params["gini"], 64)
	dt.salt = rand.Int63n(10000000000)
	dt.params.SamplingRatio, _ = strconv.ParseFloat(params["dt-sample-ratio"], 64)
}
