package hector

import(
	"flag"
	"fmt"
	"strconv"
)

func GetClassifier(method string) Classifier {
	var classifier Classifier
		
	if method == "lr"{
		classifier = &(LogisticRegression{})
	} else if method == "ftrl" {
		classifier = &(FTRLLogisticRegression{})
	} else if method == "ep" {
		classifier = &(EPLogisticRegression{})
	} else if method == "rdt" {
		classifier = &(RandomDecisionTree{})
	} else if method == "cart" {
		classifier = &(CART{})	
	} else if method == "cart-regression" {
		classifier = &(RegressionTree{})	
	} else if method == "rf" {
		classifier = &(RandomForest{})	
	} else if method == "fm" {
		classifier = &(FactorizeMachine{})	
	} else if method == "sa" {
		classifier = &(SAOptAUC{})	
	} else if method == "gbdt" {
		classifier = &(GBDT{})	
	} else if method == "svm" {
		classifier = &(SVM{})	
	} else if method == "l1vm" {
		classifier = &(L1VM{})	
	} else {
		classifier = &(LogisticRegression{})
	}
	return classifier
}

func PrepareParams() (string, string, string, string, map[string]string){
	params := make(map[string]string)
	train_path := flag.String("train", "train.tsv", "path of training file")
	test_path := flag.String("test", "test.tsv", "path of testing file")
	pred_path := flag.String("pred", "", "path of pred file")
	output := flag.String("output", "", "output file path")
	learning_rate := flag.String("learning-rate", "0.01", "learning rate")
	regularization := flag.String("regularization", "0.01", "regularization")
	alpha := flag.String("alpha", "0.1", "alpha of ftrl")
	beta := flag.String("beta", "1", "beta of ftrl")
	c := flag.String("c", "1", "C in svm")
	e := flag.String("e", "0.01", "stop threshold")
	lambda1 := flag.String("lambda1", "0.1", "lambda1 of ftrl")
	lambda2 := flag.String("lambda2", "0.1", "lambda2 of ftrl")
	tree_count := flag.String("tree-count", "10", "tree count in rdt/rf")
	feature_count := flag.String("feature-count", "-1", "feature count in rdt/rf")
	gini := flag.String("gini", "0.5", "gini threshold, between (0, 0.5]")
	min_leaf_size := flag.String("min-leaf-size", "10", "min leaf size in dt")
	max_depth := flag.String("max-depth", "10", "max depth of dt")
	factors := flag.String("factors", "10", "factor number in factorized machine")
	steps := flag.Int("steps", 1, "steps before convergent")
	global := flag.Int64("global", -1, "feature id of global bias")
	method := flag.String("method", "lr", "algorithm name")
	cv := flag.Int("cv", 7, "cross validation folder count")
	
	flag.Parse()
	fmt.Println(*train_path)
	fmt.Println(*test_path)
	fmt.Println(*method)
	params["learning-rate"] = *learning_rate
	params["regularization"] = *regularization
	params["alpha"] = *alpha
	params["beta"] = *beta
	params["lambda1"] = *lambda1
	params["lambda2"] = *lambda2
	params["tree-count"] = *tree_count
	params["feature-count"] = *feature_count
	params["max-depth"] = *max_depth
	params["min-leaf-size"] = *min_leaf_size
	params["steps"] = strconv.FormatInt(int64(*steps), 10)
	params["global"] = strconv.FormatInt(*global, 10)
	params["gini"] = *gini
	params["factors"] = *factors
	params["output"] = *output
	params["c"] = *c
	params["e"] = *e
	params["cv"] = strconv.FormatInt(int64(*cv), 10)
	
	fmt.Println(params)
	return *train_path, *test_path, *pred_path, *method, params	
}