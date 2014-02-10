package dt

import (
	"bufio"
	"container/list"
	"github.com/xlvector/hector/core"
	"io/ioutil"
	"os"
	"sort"
	"strconv"
)

type RegressionTree struct {
	tree   Tree
	params CARTParams
}

func (self *RegressionTree) SaveModel(path string) {
	ioutil.WriteFile(path, self.tree.ToString(), 0600)
}

func (self *RegressionTree) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()
	text := ""
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text += scanner.Text() + "\n"
	}
	self.tree.FromString(string(text))
}

func (dt *RegressionTree) GoLeft(sample *core.MapBasedSample, feature_split core.Feature) bool {
	value, ok := sample.Features[feature_split.Id]
	if ok && value >= feature_split.Value {
		return true
	} else {
		return false
	}
}

func (dt *RegressionTree) GetElementFromQueue(queue *list.List, n int) []*TreeNode {
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

func (dt *RegressionTree) FindBestSplit(samples []*core.MapBasedSample, node *TreeNode, select_features map[int64]bool) {
	feature_weight_labels := make(map[int64]*core.FeatureGoalDistribution)
	sum_total := 0.0
	sum_total2 := 0.0
	count_total := 0.0
	for _, k := range node.samples {
		sum_total += samples[k].Prediction
		sum_total2 += samples[k].Prediction * samples[k].Prediction
		count_total += 1.0
	}

	feature_sum_right := core.NewVector()
	feature_sum_right2 := core.NewVector()
	feature_count_right := core.NewVector()

	for _, k := range node.samples {
		for fid, fvalue := range samples[k].Features {
			feature_count_right.AddValue(fid, 1.0)
			feature_sum_right.AddValue(fid, samples[k].Prediction)
			feature_sum_right2.AddValue(fid, samples[k].Prediction*samples[k].Prediction)
			_, ok := feature_weight_labels[fid]
			if !ok {
				feature_weight_labels[fid] = core.NewFeatureGoalDistribution()
			}
			feature_weight_labels[fid].AddWeightGoal(fvalue, samples[k].Prediction)
		}
	}

	min_vari := 1e20
	node.feature_split = core.Feature{Id: -1, Value: 0}
	for fid, distribution := range feature_weight_labels {
		sort.Sort(distribution)
		split, vari := distribution.BestSplitByVariance(sum_total-feature_sum_right.GetValue(fid),
			sum_total2-feature_sum_right2.GetValue(fid),
			count_total-feature_count_right.GetValue(fid),
			feature_sum_right.GetValue(fid),
			feature_sum_right2.GetValue(fid),
			feature_count_right.GetValue(fid))
		if min_vari > vari {
			min_vari = vari
			node.feature_split.Id = fid
			node.feature_split.Value = split
		}
	}
}

func (dt *RegressionTree) AppendNodeToTree(samples []*core.MapBasedSample, node *TreeNode, queue *list.List, tree *Tree, select_features map[int64]bool) {
	if node.depth >= dt.params.MaxDepth {
		return
	}

	dt.FindBestSplit(samples, node, select_features)

	if node.feature_split.Id < 0 {
		return
	}
	left_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: core.NewArrayVector(), sample_count: 0, samples: []int{}}
	right_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: core.NewArrayVector(), sample_count: 0, samples: []int{}}

	left_positive := 0.0
	left_total := 0.0
	right_positive := 0.0
	right_total := 0.0
	for _, k := range node.samples {
		if dt.GoLeft(samples[k], node.feature_split) {
			left_node.samples = append(left_node.samples, k)
			left_positive += samples[k].Prediction
			left_total += 1.0
		} else {
			right_node.samples = append(right_node.samples, k)
			right_positive += samples[k].Prediction
			right_total += 1.0
		}
	}
	node.samples = nil

	if len(left_node.samples) > dt.params.MinLeafSize {
		left_node.sample_count = len(left_node.samples)
		left_node.prediction.SetValue(0, left_positive/left_total)
		queue.PushBack(&left_node)
		node.left = len(tree.nodes)
		tree.AddTreeNode(&left_node)
	}

	if len(right_node.samples) > dt.params.MinLeafSize {
		right_node.sample_count = len(right_node.samples)
		right_node.prediction.SetValue(0, right_positive/right_total)
		queue.PushBack(&right_node)
		node.right = len(tree.nodes)
		tree.AddTreeNode(&right_node)
	}
}

func (dt *RegressionTree) SingleTreeBuild(samples []*core.MapBasedSample, select_features map[int64]bool) Tree {
	tree := Tree{}
	queue := list.New()
	root := TreeNode{depth: 0, left: -1, right: -1, prediction: core.NewArrayVector(), samples: []int{}}
	total := 0.0
	positive := 0.0
	for i, sample := range samples {
		root.AddSample(i)
		total += 1.0
		positive += sample.Prediction
	}
	root.sample_count = len(root.samples)
	root.prediction.SetValue(0, positive/total)

	queue.PushBack(&root)
	tree.AddTreeNode(&root)
	for {
		nodes := dt.GetElementFromQueue(queue, 10)
		if len(nodes) == 0 {
			break
		}

		for _, node := range nodes {
			dt.AppendNodeToTree(samples, node, queue, &tree, select_features)
		}
	}
	return tree
}

func (dt *RegressionTree) PredictBySingleTree(tree *Tree, sample *core.MapBasedSample) (*TreeNode, string) {
	path := ""
	node := tree.GetNode(0)
	path += node.ToString()
	for {
		if dt.GoLeft(sample, node.feature_split) {
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

func (dt *RegressionTree) Train(dataset *core.DataSet) {
	samples := []*core.MapBasedSample{}
	for _, sample := range dataset.Samples {
		msample := sample.ToMapBasedSample()
		samples = append(samples, msample)
	}
	dt.tree = dt.SingleTreeBuild(samples, nil)
}

func (dt *RegressionTree) Predict(sample *core.Sample) float64 {
	msample := sample.ToMapBasedSample()
	node, _ := dt.PredictBySingleTree(&dt.tree, msample)
	return node.prediction.GetValue(0)
}

func (dt *RegressionTree) Init(params map[string]string) {
	dt.tree = Tree{}
	min_leaf_size, _ := strconv.ParseInt(params["min-leaf-size"], 10, 32)
	max_depth, _ := strconv.ParseInt(params["max-depth"], 10, 32)

	dt.params.MinLeafSize = int(min_leaf_size)
	dt.params.MaxDepth = int(max_depth)
	dt.params.GiniThreshold, _ = strconv.ParseFloat(params["gini"], 64)
}
