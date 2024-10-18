package main

import (
	"bufio"
	"github.com/disintegration/imaging"
	"github.com/zenoda/imgview"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"os"
	"path"
	"strconv"
	"strings"
)

const (
	ImageSize = 448
)

type RectMask struct {
	image.Rectangle
}

func (r *RectMask) RGBA64At(x, y int) color.RGBA64 {
	if x <= r.Min.X+1 || y <= r.Min.Y+1 || x >= r.Max.X-2 || y >= r.Max.Y-2 {
		return color.RGBA64{A: 0xffff}
	}
	return color.RGBA64{A: 0}
}

func main() {
	inputs, targets := getSamples("train")
	showResult(inputs[3], targets[3])
}

func showResult(input []float64, result []float64) {
	img := image.NewRGBA(image.Rect(0, 0, ImageSize, ImageSize))
	for x := range ImageSize {
		for y := range ImageSize {
			r, g, b, a := input[(y*ImageSize+x)*4], input[(y*ImageSize+x)*4+1], input[(y*ImageSize+x)*4+2], input[(y*ImageSize+x)*4+3]
			img.Set(x, y, color.RGBA{R: uint8(r), G: uint8(g), B: uint8(b), A: uint8(a)})
		}
	}
	uniformImg := image.NewUniform(color.RGBA{R: 255, A: 255})
	for i := range len(result) / 5 {
		x0, y0, x1, y1 := result[i*5+1]*ImageSize-result[i*5+3]*ImageSize/2, result[i*5+2]*ImageSize-result[i*5+4]*ImageSize/2, result[i*5+1]*ImageSize+result[i*5+3]*ImageSize/2, result[i*5+2]*ImageSize+result[i*5+4]*ImageSize/2
		rect := &RectMask{image.Rect(int(x0), int(y0), int(x1), int(y1))}
		draw.DrawMask(img, img.Bounds(), uniformImg, image.Pt(0, 0), rect, image.Pt(0, 0), draw.Over)
	}
	imgview.Show(img)
}

func getSamples(sampleType string) (inputs [][]float64, targets [][]float64) {
	var imgRootPath, labelRootPath string
	switch sampleType {
	case "train":
		imgRootPath = "../dataset/images/train/"
		labelRootPath = "../dataset/labels/train/"
		break
	case "test":
		imgRootPath = "../dataset/images/val/"
		labelRootPath = "../dataset/labels/val/"
		break
	}
	imgDir, err := os.Open(imgRootPath)
	if err != nil {
		panic(err)
	}
	dirEntries, err := imgDir.ReadDir(-1)
	if err != nil {
		panic(err)
	}
	for _, dirEntry := range dirEntries {
		name := strings.Split(dirEntry.Name(), ".")
		imgPath := path.Join(imgRootPath, dirEntry.Name())
		labelPath := path.Join(labelRootPath, name[0]+".txt")
		input := readImg(imgPath)
		target := readLabel(labelPath)
		inputs = append(inputs, input)
		targets = append(targets, target)
	}
	return
}

func readLabel(labelPath string) (target []float64) {
	f, err := os.Open(labelPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		cells := strings.Split(scanner.Text(), " ")
		for _, cell := range cells {
			data, err := strconv.ParseFloat(cell, 64)
			if err != nil {
				panic(err)
			}
			target = append(target, data)
		}
	}
	return
}

func readImg(imgPath string) (input []float64) {
	f, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	img, err := jpeg.Decode(f)
	if err != nil {
		panic(err)
	}
	img = imaging.Resize(img, ImageSize, ImageSize, imaging.NearestNeighbor)
	for y := range img.Bounds().Dy() {
		for x := range img.Bounds().Dx() {
			pixel := img.At(x, y)
			r, g, b, a := pixel.RGBA()
			input = append(input, float64(r), float64(g), float64(b), float64(a))
		}
	}
	return
}
