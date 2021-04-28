const fs = require('fs');
const path = require("path");
const { exit } = require('process');

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

    // adds data to the instance
    // data should be of the form
    // [{input: [int], output: int}]
    // an array of objects, each object having an
    // input, which is an array of ints,
    // and an output, which is an int
    addData(data){
        this.data = data;
        this.maxRange = data[0].input.length;
    }

    // the main loop
    // generates a population,
    // then tests it, until the error rate is reached
    train(errorRate){
        var generations = 0;
        console.log("Thinking...");
        var error = (this.population.size > 0) ? this.population[0].error : 100;
        // while the error of the current best neural network
        // exceeds the specified target error, continue looping
        while (error > errorRate){
            // generate the population
            this.regeneratePopulation();
            // test the population against the data
            // this will also sort the population
            this.testPopulation();
            // remember the old error rate for comparison
            var oldError = error;
            // the new error rate is the error rate of the best neural network
            error = this.population[0].error;
            generations += 1;
            // scale mutation rate based on how incorrect the current best nn is
            this.mutationRate = error * 10;
            if (oldError > error){
                console.log("New lowest error: ", error, " at generation: ", generations);
            }
            // console.log("Current error of best neural network: ", error);
            // if (oldMutationRate != this.mutationRate){
            //     console.log("Updating mutation rate to: ", this.mutationRate);
            // }
            // console.log("Starting generation: ", generations);
        }
        // print the weights of the best model
        // if one is ever found which performs well enough
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
                    // generate a new NN based on the index chosen
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
        if (progenitor == null){ // if this is just generating a random nn rather than copying one
            var range = Math.floor(Math.random() * (this.maxRange - 1)) + 1;
            child = {
                // need to + 1 to the range to allow for a bias in the input layer
                weights: [this.generateMatrix(range + 1, 1)],
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
            // change the range (this also requires changing the first layer)
            var newValues = this.changeRange(clone.range, clone.weights[0], clone.maxRange);
            clone.range = newValues.newRange;
            clone.weights[0] = newValues.newMatrix;
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
        second = this.generateMatrix(p, m);
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
    changeRange(currentRange, firstMatrix, maxRange){
        var roll = Math.random() * 100;
        var newRange = currentRange;
        var newMatrix = this.clone(firstMatrix);
        if (currentRange > 1 && roll > 50 || currentRange == maxRange){
            // decrease range
            newRange -= 1;
            newMatrix = newMatrix.slice(0, -1);
        } else {
            //increase range
            newRange +=1;
            var newRow = this.generateRow(newMatrix[0].length);
            newMatrix.push(newRow);
        }
        return {
            newRange,
            newMatrix
        }
    }

    // pads out a row of weights with more weights
    // inserts new weights randomly
    padRow(row, number){
        var clone = this.clone(row);
        for (let i = 0; i < number; i++){
            var index = Math.floor(Math.random() * (clone.length - 1));
            clone.splice(index, 0, Math.random());
        }
        return clone;
    }

    // randomly removes a number of weights from a row
    trimRow(row, number){
        var clone = this.clone(row);
        for (let i = 0; i < number; i++){
            var index = Math.floor(Math.random() * (clone.length - 1));
            clone.splice(index, 1);
        }
        return clone;
    }

    // generates a matrix with random weights
    generateMatrix(rowSize, colSize){
        var rtn = [];
        for (let i = 0; i < rowSize; i++){
            var row = this.generateRow(colSize);
            rtn.push(row);
        }
        return rtn;
    }

    // generates a row of weights for a matrix
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
            var prediction = Math.round(this.propagate(this.data[i].input, nn.range, nn.weights));
            // compare it to the actual outcome
            var currentError = Math.abs(prediction - this.data[i].output);
            // add any error to the overall error
            error += currentError;
            // random weights on huge datasets can have a summed error rate
            // larger than js can handle, in which case it will return NaN
            if (isNaN(error)){
                return (Number.MAX_SAFE_INTEGER/this.data.length);
            }
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
            // update the currentValues array by multiplying through the next layer of weights
            currentValues = this.multiply(currentValues, weights[i]);
        }
        // if the final array has more than 1 value, something has gone wrong
        // the final array should be the output, which will be 1 value
        if (currentValues.length != 1){
            var error = "Error: Neural network collapsed into more than one output value";
            console.log(error);
            throw new Error(error);
        } else {
            // if there is one value, return that value
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
        // sort the population based on error rate (ascending)
        this.population.sort((a, b) => {
            if (a.error < b.error){
                return -1;
            } else if (a.error > b.error){
                return 1;
            } else {
                // if the error rate is the same,
                // prefer the matrix with less range
                // this will prevent networks growing arbitrarily large
                // it is also preferable to look at a smaller range to speed up the processing
                if (a.range < b.range){
                    return -1;
                } else if (a.range > b.range){
                    return 1;
                } else {
                    return 0;
                }
            }
        })
    }

    // Multiplies matricies A and B
    // Does not mutate A or B
    // Not optimised ? Naive implementation
    // A is a n-size array of current values
    // B is a n by m array of weights,
    // C is the output, a m-size array of new current values
    // As such, this isn't universal matrix multiplication
    // It is array by matrix multiplication
    multiply(a, b){
        if (a.length != b.length){
            console.log("Error while multiplying matricies a: ", a, ", and b: ", b, "\nLength of a: ", a.length, "\nLength of b: ", b.length);
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

    // activation functions

    sigmoid(t) {
        return 1/(1+Math.pow(Math.E, -t));
    }

    reLU(t){
        return Math.max(0, t);
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
        // get absolute rather than relative paths
        const absolute = path.join(dir, file);
        // if the 'file' is a directory, recurse on it
        if (fs.statSync(absolute).isDirectory()) {
            files = files.concat(getAllFiles(absolute));
        } else { // otherwise, add it to the list
            files.push(absolute);
        }
    });
    return files;
}

// formats data from a directory of files into the format expected by the neural network
// this will break if there is any change to the format of the data files
// input type is [{filepath: String, label: int}]
// an array of objects, each object points to a directory and has an output label
// this allows for reading in both shots and nonshots
// reading in just shots or just nonshots won't produce anything worthwhile
// because the neural network will optimise for always producing 1 or 0
function formatData(groups){
    var dataArray = [];
    for (let i = 0; i < groups.length; i++){
        var filepath = groups[i].filepath;
        var label = groups[i].label;
        // if someone forgot to either add the correct directory,
        // or refactor the code to require a terminal input
        // which seems more cumbersome to me
        // let them know of their mistake
        if (!fs.existsSync(filepath)){
            console.log("Could not find directory: ", filepath, "\nPlease change the code to include an existing directory.");
            process.exit(1);
        }
        // get the names of all files in the directory and subdirectories
        var allFiles = getAllFiles(filepath);
        // maintain a dictionary of data
        // this will safeguard against the possibility of an axis of data missing from a particular shot
        var dataDictionary = {};
        allFiles.forEach(filename => {
            if (filename.split('.').pop() == "csv"){ // make sure the file is .csv extension
                // read in the file
                var fileContents = fs.readFileSync(filename, "ascii");
                // split the file into an array and convert the string numbers to ints
                var inputArray = fileContents.split("\n").map(x=> Math.abs(parseInt(x)));
                // if the file contains x axis data
                if (filename.includes("X__AXIS")){
                    var key = filename.replace("X__AXIS", "");
                    // either create the entry if it doesn't exist
                    if (!dataDictionary.hasOwnProperty(key)){
                        dataDictionary.key = {
                            x: inputArray,
                            y: null,
                            z: null,
                        }
                    } else {
                        // or set the x property of the entry
                        dataDictionary.key.x = inputArray;
                    }
                } else if (filename.includes("Y__AXIS")){ // same logic for Y axis
                    var key = filename.replace("Y__AXIS", "");
                    if (!dataDictionary.hasOwnProperty(key)){
                        dataDictionary.key = {
                            x: null,
                            y: inputArray,
                            z: null,
                        }
                    } else {
                        dataDictionary.key.y = inputArray;
                    }
                } else if (filename.includes("Z__AXIS")){ // same logic for Z axis
                    var key = filename.replace("Z__AXIS", "");
                    if (!dataDictionary.hasOwnProperty(key)){
                        dataDictionary.key = {
                            x: null,
                            y: null,
                            z: inputArray,
                        }
                    } else {
                        dataDictionary.key.z = inputArray;
                    }
                } else {
                    console.log("It looks like naming conventions have changed.\nThis program looks for files with X__AXIS or similar in the filename, please update the code.");
                    process.exit(1);
                }
            }
        });
        // get the keys of the data dictionary
        // at this point, the dictionary should have a list of keys with axis stripped out
        // each key points to an object, and the object should have x, y, and z properties
        var keys = Object.keys(dataDictionary);
        for (let i = 0; i < keys.length; i++){
            // make sure this entry has x, y, and z properties
            if (dataDictionary[keys[i]].x != null && dataDictionary[keys[i]].y != null && dataDictionary[keys[i]].z != null){
                // add the concatenated data plus the label to the data array
                dataArray.push({
                    input: dataDictionary[keys[i]].x.concat(dataDictionary[keys[i]].y, dataDictionary[keys[i]].z),
                    output: label,
                })
            }
        }
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

// var n = alg.generateNN(null);

// var currentValues = alg.data[0].input;

// currentValues = currentValues.slice(0,n.range);
// currentValues.push(1);

// console.log("generated weights length: ", alg.generateMatrix(n.range + 1, 1).length);
// console.log("length of net's first weights array: ", n.weights[0].length);
// console.log("currentvalues length: ", currentValues.length);
// console.log("based on range: ", n.range);

// console.log("range: ", n.range, "\nweights: ", n.weights[0], "\ndata: ", currentValues,
//     "\nSize of weights: ", n.weights[0].length, "\nLength of input: ", currentValues.length);

alg.train(0.1);