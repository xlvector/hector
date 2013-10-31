package hector

import(
	"strconv"
	"math/rand"
	"math"
	"os"
	"strings"
)

type KMeans struct {
	k int
	centers []map[int64]float64
}

func (c *KMeans) Init(params map[string]string){
	center_number, _ := strconv.ParseInt(params["k"], 10, 64)
	c.k = int(center_number)
	
	c.centers = []map[int64]float64{}
}

func (c *KMeans) Distance(sample *Sample, j int) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		fj, ok := c.centers[j][feature.Id]
		if ok {
			ret += feature.Value * fj
		}	
	}
	r1 := 0.0
	r2 := 0.0
	for _, feature := range sample.Features {
		r1 += feature.Value * feature.Value
	}
	for _, fvalue := range c.centers[j] {
		r2 += fvalue * fvalue
	}
	if r1 < 1e-10 || r2 < 1e-10{
		return 0.0
	}
	ret /= math.Sqrt(r1 * r2)
	return ret
}

func (c *KMeans) NearestCenter(sample *Sample) int {
	ret := 0
	mindis := -1.0
	for i := range c.centers {
		dis := c.Distance(sample, i)
		if mindis < 0 {
			mindis = dis
			ret = i
		} else {
			if mindis > dis{
				mindis = dis
				ret = i
			}
		}
	}
	return ret
}

func (c *KMeans) AddToCenter(i int, sample Sample){
	for _, feature := range sample.Features {
		fvalue, ok := c.centers[i][feature.Id]
		if !ok {
			c.centers[i][feature.Id] = feature.Value
		} else {
			c.centers[i][feature.Id] += fvalue
		}
	}	
}

func (c *KMeans) SaveModel(output string) error {
	f, err := os.Create(output)
	if err != nil{
		return err
	}
	defer f.Close()
	for _, ci := range c.centers{
		line := ""
		for fid, fvalue := range ci{
			line += strconv.FormatInt(fid, 10)
			line += ":"
			line += strconv.FormatFloat(fvalue, 'g', 8, 64)
			line += "\t"	
		}
		line = strings.Trim(line, "\t") + "\n"
		f.WriteString(line)
	}
	return nil	
}

func (c *KMeans) Cluster(dataset DataSet){
	samples := []Sample{}
	assign := []int{}
	for sample := range dataset.Samples {
		samples = append(samples, sample)
		assign = append(assign, 0)
	}
	
	for i:= 0; i < c.k; i++{
		c.centers = append(c.centers, samples[rand.Int() % len(samples)].ToMapBasedSample().Features)
	}
	
	for step := 0; step < 20; step++{
		for i, sample := range samples {
			assign[i] = c.NearestCenter(&sample)	
		}
		
		for i:= 0; i < c.k; i++{
			c.centers[i] = make(map[int64]float64)
		}
		
		count := []float64{}
		for i := 0; i < c.k; i++{
			count = append(count, 0.0)
		}
		for i, ci := range assign{
			c.AddToCenter(ci, samples[i])
			count[ci] += 1.0	
		}
		for i, ci := range c.centers{
			for fid, fvalue := range ci{
				ci[fid] = fvalue / count[i]
			}
		}
	}
}