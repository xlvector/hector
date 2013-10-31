package hector

import(
	"flag"
	"fmt"
	"strconv"
)

func PrepareParams() (string, string, string, map[string]string){
	params := make(map[string]string)
	train_path := flag.String("train", "train.tsv", "path of training file")
	test_path := flag.String("test", "test.tsv", "path of testing file")
	learning_rate := flag.String("learning-rate", "0.01", "learning rate")
	regularization := flag.String("regularization", "0.01", "regularization")
	alpha := flag.String("alpha", "0.1", "alpha of ftrl")
	beta := flag.String("beta", "1", "beta of ftrl")
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
	fmt.Println(params)
	return *train_path, *test_path, *method, params	
}