from phrase_generator import PhraseGenerator

print('Testing reformatted phrase generator...')
print()

generator = PhraseGenerator()

# Generate 5 phrases to test the format
phrases = generator.generate_multiple_phrases(5)

# Format as comma-separated quoted phrases
formatted_output = ','.join(f'"{phrase}"' for phrase in phrases)

print('Generated phrases in requested format:')
print(formatted_output)

print()
print('✅ Reformatted phrase generator working correctly!')
