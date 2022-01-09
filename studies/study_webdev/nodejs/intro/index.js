// const fs = require("fs");
//
// fs.copyFileSync("file1.txt", "file2.txt");
const superheroes = require('superheroes');

superheroes.all;
//=> ['3-D Man', 'A-Bomb', …]

var hero = superheroes.random();
//=> 'Spider-Ham'

console.log(hero)