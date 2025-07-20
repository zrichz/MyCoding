void setup() {
  // Open the input file
  String[] lines = loadStrings("input.txt");

  // Define arrays for colours and materials
  String[] arr_colour = {"red", "blue", "green", "yellow", "black"}; // Array of colours
  String[] arr_material = {"wood", "metal", "plastic", "glass", "stone"}; // Array of materials

  // StringBuilder to accumulate new content
  StringBuilder newContent = new StringBuilder();

  // Process each line of the input file
  for (String line : lines) {
    println("Processing line: " + line); // Debug: Print the current line
    // Split the line into words and punctuation, preserving both
    String[] words = splitWithDelimiters(line);
    println("Split words: " + join(words, ", ")); // Debug: Print the split words

    // Scan through the line word by word
    for (int i = 0; i < words.length; i++) {
      String word = words[i];
      println("Processing word: " + word); // Debug: Print the current word

      // Check if the word starts with '#' (indicating a wildcard)
      if (word.startsWith("#") && word.length() > 1) {
        // Extract the word immediately following the hash
        String Wildcard = word.substring(1); // Remove the '#' prefix
        println("Found wildcard: " + Wildcard); // Debug: Print the wildcard

        // Replace wildcard with a random item from the relevant array
        if (Wildcard.equalsIgnoreCase("colour")) {
          words[i] = arr_colour[int(random(arr_colour.length))]; // Replace with random colour
          println("Replaced with: " + words[i]); // Debug: Print the replacement
        } else if (Wildcard.equalsIgnoreCase("material")) {
          words[i] = arr_material[int(random(arr_material.length))]; // Replace with random material
          println("Replaced with: " + words[i]); // Debug: Print the replacement
        } else {
          println("No matching wildcard found for: " + Wildcard); // Debug: No match
        }
      }

      // Append the processed word to the new content
      newContent.append(words[i]); // Add the word (or replacement) to the output
    }

    // Add a newline character after processing each line
    newContent.append("\n");
  }

  // Save the new content to the output file
  saveStrings("data/output.txt", newContent.toString().split("\n")); // Write output to file
  println("Final content: " + newContent); // Debug: Print the final content
  println("File saved as data/output.txt"); // Confirm file save
}

// Custom method to split a string while keeping the delimiters
String[] splitWithDelimiters(String input) {
  ArrayList<String> result = new ArrayList<String>(); // List to store split parts
  StringBuilder word = new StringBuilder(); // Temporary storage for building words

  for (char c : input.toCharArray()) {
    if (Character.isLetterOrDigit(c)) {
      word.append(c); // Add character to the current word if it's alphanumeric
    } else {
      if (word.length() > 0) {
        result.add(word.toString()); // Add the completed word to the result
        word.setLength(0); // Reset the word builder
      }
      result.add(Character.toString(c)); // Add the delimiter as a separate entry
    }
  }
  if (word.length() > 0) {
    result.add(word.toString()); // Add the last word if any
  }

  return result.toArray(new String[0]); // Convert the list to an array and return
}
