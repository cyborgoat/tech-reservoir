const numbers = [1, 2, 3];
const newNumbers = [...numbers, 4];
console.log(newNumbers);

const person = {
    name: 'Max'
};

const newPerson = {
    ...person,
    age: 28
};

console.log(newPerson.name)

const filter = (...args) => {
    return args.filter(el => el === 1);
}

console.log(filter(1, 2, 3, 4, 5));

//
[a, b] = ['Hello', 'Max'];
console.log(a);
console.log(b);

// shallow copy & deep copy
const person1 = {
    name: "max"
};

const secondPerson = person1;
const thirdPerson = {
    ...person1
};
person1.name = "Manu";
console.log(person1.name)
console.log(secondPerson.name)
console.log(thirdPerson.name)

// for es6/babel
// const numbers = [1, 2, 3];
// [a, , b] = numbers;
// console.log(a);
// console.log(b);