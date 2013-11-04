package hector

import (
	"math/rand"
	"sync"
	"time"
	"strconv"
)

type TreeNode struct {
	left, right, depth int
	prediction         float64
	sample_count		int
	samples            []int
	feature_split      Feature
}

func (t *TreeNode) ToString() string {
	return strconv.FormatInt(t.feature_split.Id, 10) + ":" + strconv.FormatFloat(t.feature_split.Value, 'g', 3, 64)
}

func (t *TreeNode) AddSample(k int) {
	t.samples = append(t.samples, k)
}

type Tree struct {
	nodes []*TreeNode
}

func (t *Tree) AddTreeNode(n *TreeNode) {
	t.nodes = append(t.nodes, n)
}

func (t *Tree) Size() int {
	return len(t.nodes)
}

func (t *Tree) GetNode(i int) *TreeNode {
	return t.nodes[i]
}

type RDTParams struct {
	TreeCount   int
	MinLeafSize int
}

type RandomDecisionTree struct {
	trees []*Tree
	params RDTParams
}

func (rdt *RandomDecisionTree) GoLeft(sample *MapBasedSample, feature_split Feature) bool {
	value, ok := sample.Features[feature_split.Id]
	if ok && value >= feature_split.Value {
		return true
	} else {
		return false
	}
}

func (rdt *RandomDecisionTree) GetElementFromQueue(queue chan *TreeNode, n int) []*TreeNode {
	ret := []*TreeNode{}
	for i := 0; i < n; i++ {
		if len(queue) == 0 {
			time.Sleep(1e9)
			if len(queue) == 0 {
				break
			}
		}
		node := <-queue
		ret = append(ret, node)
	}
	return ret
}

func (rdt *RandomDecisionTree) AppendNodeToTree(samples []*MapBasedSample, node *TreeNode, queue chan *TreeNode, tree *Tree, feature_splits []Feature) {
	positive := 0.0
	total := 0.0
	for _, k := range node.samples {
		positive += samples[k].LabelDoubleValue()
		total += 1.0
	}
	node.prediction = positive / total
	node.sample_count = int(total)

	if node.depth >= len(feature_splits) {
		return
	}
	node.feature_split = feature_splits[node.depth]
	left_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: -1, sample_count: 0, samples: []int{}}
	right_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: -1, sample_count: 0, samples: []int{}}

	for _, k := range node.samples {
		positive += samples[k].LabelDoubleValue()
		total += 1.0
		if rdt.GoLeft(samples[k], feature_splits[node.depth]) {
			left_node.samples = append(left_node.samples, k)
		} else {
			right_node.samples = append(right_node.samples, k)
		}
	}
	node.samples = nil

	if len(left_node.samples) > rdt.params.MinLeafSize {
		queue <- &left_node
		node.left = len(tree.nodes)
		tree.AddTreeNode(&left_node)
	}

	if len(right_node.samples) > rdt.params.MinLeafSize {
		queue <- &right_node
		node.right = len(tree.nodes)
		tree.AddTreeNode(&right_node)
	}
}

func (rdt *RandomDecisionTree) SingleTreeBuild(samples []*MapBasedSample, feature_splits []Feature) Tree {
	tree := Tree{}
	queue := make(chan *TreeNode, 4096)
	root := TreeNode{depth: 0, left: -1, right: -1, prediction: -1, samples: []int{}}
	for i, _ := range samples {
		root.AddSample(i)
	}

	queue <- &root
	tree.AddTreeNode(&root)
	for {
		nodes := rdt.GetElementFromQueue(queue, 10)
		if len(nodes) == 0 {
			break
		}

		for _, node := range nodes {
			rdt.AppendNodeToTree(samples, node, queue, &tree, feature_splits)
		}
	}
	return tree
}

func (rdt *RandomDecisionTree) PredictBySingleTree(tree *Tree, sample *MapBasedSample) *TreeNode {
	node := tree.GetNode(0)
	for {
		if rdt.GoLeft(sample, node.feature_split) {
			if node.left >= 0 && node.left < tree.Size() {
				node = tree.GetNode(node.left)
			} else {
				return node
			}
		} else {
			if node.right >= 0 && node.right < tree.Size() {
				node = tree.GetNode(node.right)
			} else {
				return node
			}
		}
	}
	return node
}

func (rdt *RandomDecisionTree) RandomShuffle(features []Feature){
	for i := range features {
	    j := rand.Intn(i + 1)
	    features[i], features[j] = features[j], features[i]
	}
}

func (rdt *RandomDecisionTree) Train(dataset * DataSet) {
	samples := []*MapBasedSample{}
	for _, sample := range dataset.Samples{
		samples = append(samples, sample.ToMapBasedSample())
	}
	forest := make(chan *Tree, rdt.params.TreeCount)
	var wait sync.WaitGroup
	wait.Add(rdt.params.TreeCount)
	for k := 0; k < rdt.params.TreeCount; k++ {
		go func() {
			feature_split := []Feature{}
			for j:= 0; j < 5; j++{
				m := rand.Int() % len(samples)
				random_sample := samples[m]
				
				for fid, fvalue := range random_sample.Features {
					feature_split = append(feature_split, Feature{Id: fid, Value: fvalue})
				}
			}
			rdt.RandomShuffle(feature_split)
			tree := rdt.SingleTreeBuild(samples, feature_split)
			forest <- &tree
			wait.Done()
		}()
	}
	wait.Wait()
	close(forest)
	for tree := range forest{
		rdt.trees = append(rdt.trees, tree)
	}
}

func (rdt *RandomDecisionTree) Predict(sample * Sample) float64 {
	ret := 0.0
	total := 0.0
	msample := sample.ToMapBasedSample()
	for _,tree := range rdt.trees{
		node := rdt.PredictBySingleTree(tree, msample)
		ret += node.prediction
		total += 1.0
	}	
	return ret / total
}

func (rdt *RandomDecisionTree) Init(params map[string]string) {
	rdt.trees = []*Tree{}
	min_leaf_size, _ := strconv.ParseInt(params["min-leaf-size"], 10, 32)
	tree_count, _ := strconv.ParseInt(params["tree-count"], 10, 32)
	
	rdt.params.MinLeafSize = int(min_leaf_size)
	rdt.params.TreeCount = int(tree_count)
}

