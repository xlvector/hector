package util


func CombineFeatures(fids []int64) int64{
	ret := int64(0)
	
	for _, fid := range fids{
		ret *= 601840361
		ret += fid
	}	
	if ret < 0 {
		ret *= -1
	}
	return ret
}

func Hash(str string) int64 {
	h := int64(0)

	for _, ch := range str {
		h *= 601840361
		h += int64(ch)
	}
	if h < 0 {
		return -1 * h;
	}
	return h
}