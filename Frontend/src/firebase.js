// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyB7bampULeRraZvREojHpb7rYYnu75dTp0",
  authDomain: "newsorchestra.firebaseapp.com",
  projectId: "newsorchestra",
  storageBucket: "newsorchestra.firebasestorage.app",
  messagingSenderId: "639748979487",
  appId: "1:639748979487:web:414c2f7710ea2c0b224e2a",
  measurementId: "G-L677F3DSDF"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);