const fs = require('fs');
const path = require("path");

class GeneticAlgorithm{
    // inputs is a n by m array (n samples, m dimensions of each sample, not including a bias)
    // outputs is a 1 by n array (collapse each sample into a single value)
    constructor(){
        this.population = [];
        this.data = null;
        this.mutationRate = 10;
        this.survivorRate = 20;
        this.populationSize = 100;
        this.maxParents = 10;
    }

    addData(data){
        this.data = data;
        this.maxRange = data[0].input.length;
    }

    // the main loop
    // generates a population,
    // then tests it, until the error rate is reached
    train(errorRate){
        var error = (this.population.size > 0) ? this.population[0].error : 100;
        while (error > errorRate){
            this.regeneratePopulation();
            this.testPopulation();
            error = this.population[0].error;
        }
        console.log("Best model: ", this.population[0].weights);
    }

    // fills the population up to populationSize by combining the fittest nns
    // mutationRate chance of a random nn spawning
    // if population is empty, will create populationSize new ones
    regeneratePopulation(){
        if (this.population.length == 0){
            for (let i = 0; i < this.populationSize; i++){
                var newNN = this.generateNN(null);
                this.population.push(newNN);
            }
        } else {
            // remove the weakest nns
            this.population = this.population.slice(0,this.survivorRate);
            // generate more
            while (this.population.length < this.populationSize){
                var roll = Math.floor(Math.random() * 100);
                if (roll < this.mutationRate){
                    // spawn a random NN
                    var newNN = this.generateNN(null);
                    this.population.push(newNN);
                } else {
                    // generates a number in a probability distribution favouring lower numbers
                    var index = Math.floor(Math.random() * Math.random() * this.survivorRate);
                    var newNN = this.generateNN(this.population[index]);
                    this.population.push(newNN);
                }
            }
        }


    }

    // generates a random ROI based on range
    // currently unused
    // generateROI(){
    //     var one = Math.floor(Math.random() * this.maxRange);
    //     var other = Math.floor(Math.random() * this.maxRange);
    //     return {
    //         min: Math.min(one, other),
    //         max: Math.max(one, other),
    //     }
    // }

    // generates a new nn based on a successful nn
    generateNN(progenitor){
        var child;
        if (progenitor == null){
            var range = Math.floor(Math.random() * (this.maxRange - 1)) + 1;
            child = {
                weights: [this.generateWeights(range + 1, 1)],
                range
            };
        } else {
            child = this.mutateNN(progenitor);
        }
        return child;
    }

    mutateNN(nn){
        var clone = this.clone(nn);
        // roll for changing the range
        var rangeRoll = Math.random() * 100;
        if (rangeRoll < this.mutationRate){
            // change the range
            var newValues = this.changeRange(clone.range, clone.weights[0]);
            clone.range = newValues.newRange;
            clone[0] = newValues.newMatrix;
        }
        // for each layer, roll for adding a layer after it, or removing the layer
        for (let i = 0; i < clone.length; i++){
            var layerRoll = Math.random() * 100;
            if (layerRoll < this.mutationRate){
                if (i > 0 && layerRoll < (this.mutationRate / 2)){
                    // remove this layer
                    var newLayer = this.removeLayer(clone[i-1], clone[1]);
                    clone.splice(i - 1, 2, newLayer);
                } else {
                    // add a layer
                    var newLayers = this.addLayer(clone[i]);
                    clone.splice(i, 1, newLayers.first, newLayers.second);
                }
            }

        }
        return clone;
    }

    // deep copy of an object
    clone(o){
        return JSON.parse(JSON.stringify(o));
    }


    // turns a n by m matrix into a n by p matrix and a p by m matrix
    addLayer(matrix){
        var clone = this.clone(matrix);
        var m = matrix[0].length;
        var p = Math.floor(Math.random() * (this.range - 1)) + 1;
        var first;
        var second;
        // if m < p we need to add elements to each row of the matrix
        if (m < p){
            first = clone.map(x => this.padRow(x, (p-m)));
        } else if (m > p){
            // if m > p we need to remove elements from each row of the matrix
            first = clone.map(x => this.trimRow(x, (m-p)));
        }
        second = this.generateWeights(p, m);
        return {
            first, second
        }
    }

    // changes an n by m matrix and an m by p matrix into an n by p matrix
    removeLayer(first, second){
        var m = first[0].length;
        var p = second[0].length;
        var newMatrix = this.clone(first);
        // if the length of each row is longer than it should be, trim each row
        if (m > p){
            newMatrix = newMatrix.map(x => this.trimRow(x, (m-p)));
        } else {
            // otherwise pad each row
            newMatrix = newMatrix.map(x => this.padRow(x, (p-m)));
        }
        return newMatrix;
    }


    // changes the range of inputs the nn looks at
    changeRange(currentRange, firstMatrix){
        var roll = Math.random() * 100;
        var newRange = currentRange;
        var newMatrix = this.clone(firstMatrix);
        if (currentRange > 1 && roll > 50){
            // decrease range
            newRange -= 1;
            newMatrix = newMatrix.map(x => this.trimRow(x, 1));
        } else {
            //increase range
            newRange +=1;
            newMatrix = newMatrix.map(x=> this.padRow(x, 1));
        }
        return {
            newRange,
            newMatrix
        }
    }

    padRow(row, number){
        var clone = this.clone(row);
        for (let i = 0; i < number; i++){
            var index = Math.floor(Math.random() * (clone.length - 1));
            clone.splice(index, 0, Math.random());
        }
        return clone;
    }

    trimRow(row, number){
        var clone = this.clone(row);
        for (let i = 0; i < number; i++){
            var index = Math.floor(Math.random() * (clone.length - 1));
            clone.splice(index, 1);
        }
        return clone;
    }

    generateWeights(rowSize, colSize){
        var rtn = [];
        for (let i = 0; i < rowSize; i++){
            var row = this.generateRow(colSize);
            rtn.push(row);
        }
        return rtn;
    }

    generateRow(rowLength){
        var rtn = [];
        for (let i = 0; i < rowLength; i++){
            rtn.push(Math.random());
        }
        return rtn;
    }

    // tests a neural network and returns the error rate
    testNeuralNetwork(nn){
        // keep track of overall error rate
        var error = 0;
        // keep track of all processed data points
        var total = 0;
        // for each data point in the inputs
        for (let i = 0; i < this.data.length; i++){
            // get the predicted outcome
            var prediction = this.propagate(this.data[i].input, nn.range, nn.weights);
            // compare it to the actual outcome
            var currentError = Math.abs(prediction - this.data[i].output);
            // add any error to the overall error
            error += currentError;
            // increment the total
            total += 1;
        }
        return (error/total);
    }

    // propagate pushes an input through a nn and returns the output
    propagate(input, range, weights){
        // get the correct range and add a bias
        var currentValues = input.slice(0,range);
        currentValues.push(1);
        for (let i = 0; i < weights.length; i++){
            currentValues = this.multiply(currentValues, weights[i])
        }
        if (currentValues.length != 1){
            var error = "Error: Neural network collapsed into more than one output value";
            console.log(error);
            throw new Error(error);
        } else {
            return currentValues[0];
        }
    }

    // tests all of the nns in the population and sorts them
    testPopulation(){
        for (let i = 0; i < this.population.length; i++){
            // get the error rate for this nn
            var error = this.testNeuralNetwork(this.population[i]);
            // update the nn with the error rate
            this.population[i].error = error;
        }
        // sort the population based on error rate
        this.population.sort((a, b) => {
            if (a.error < b.error){
                return -1;
            } else if (a.error > b.error){
                return 1;
            } else {
                if (a.range < b.range){
                    return -1;
                } else if (a.range > b.range){
                    return 1;
                } else{
                    return 0;
                }
            }
        })
    }

    // Multiplies matricies A and B
    // Does not mutate A or B
    // Not optimised 
    // A is a n-size array of current values
    // B is a n by m array of weights,
    // C is the output, a m-size array of new current values
    multiply(a, b){
        if (a.length != b.length){
            console.log("Error while multiplying matricies a: ", a, ", and b: ", b);
            throw new Error();
        }
        const n = a.length;
        const m = b[0].length;
        var C = [];
        for (let colB = 0; colB < m; colB++){
            var total = 0;
            for (let cell = 0; cell < n; cell++){
                total += (a[cell] * b[cell][colB]);
            }
            // Push a value onto C
            // C will have m elements
            //C.push(this.sigmoid(total));
            C.push(total);
        }
        return C;
    }

    sigmoid(t) {
        return 1/(1+Math.pow(Math.E, -t));
    }

    // Prints the model for the fittest nn
    print(){
        var best = this.population[0];
        var str = "";
        str += ("Range: " + best.range);
        for (let i = 0; i < best.weights; i++){
            str += ("\nLayer: " + best.weights[i]);
        }
        console.log(str);
    }
}

function getAllFiles(dir) {
    var files = [];
    fs.readdirSync(dir).forEach(file => {
        const absolute = path.join(dir, file);
        if (fs.statSync(absolute).isDirectory()) {
            files = files.concat(getAllFiles(absolute));
            }
        else{
            files.push(absolute);
        }
    });
    return files;
}

function formatData(groups){
    var dataArray = [];
    for (let i = 0; i < groups.length; i++){
        var filepath = groups[i].filepath;
        var label = groups[i].label;
        var allFiles = getAllFiles(filepath);
        allFiles.forEach(filename => {
            if (filename.split('.').pop() == "csv"){
                var axis;
                if (filename.includes("X__AXIS")){
                    axis = 1;
                } else if (filename.includes("Y__AXIS")){
                    axis = 2;
                } else if (filename.includes("Z__AXIS")){
                    axis = 3;
                }
                var fileContents = fs.readFileSync(filename, "ascii");
                var inputArray = fileContents.split("\n").map(x=> parseInt(x));
                inputArray.splice(1, 0, axis);
                dataArray.push({
                    input: inputArray,
                    output: label
                });
            }
        })
    }

    return dataArray;
}

var dataStreams = [
    {filepath: "C:\\Users\\david\\Desktop\\randomrest", label: 1},
    {filepath: "C:\\Users\\david\\Desktop\\Data\\Nonshots", label: 0}
];

var data = formatData(dataStreams);

var alg = new GeneticAlgorithm();
alg.addData(data);

alg.train(0.0001);