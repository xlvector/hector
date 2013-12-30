package core

type FeatureType int

var FeatureTypeEnum = struct {
	DISCRETE_FEATURE FeatureType
	CONTINUOUS_FEATURE FeatureType
}{0, 1}

func GetFeatureType(key string) FeatureType {
	if key[0] == '#' {
		return FeatureTypeEnum.DISCRETE_FEATURE
	} else {
		return FeatureTypeEnum.CONTINUOUS_FEATURE
	}
}

type Feature struct {
	Id int64
	Value float64
}