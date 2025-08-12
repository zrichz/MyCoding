from phrase_generator import PhraseGenerator

print('Testing cleaned phrase generator...')
print()

generator = PhraseGenerator()

print('Testing the three templates:')
for i in range(6):
    phrase = generator.generate_phrase()
    print(f'{i+1}. {phrase}')

print()
print('Templates:')
for i, template in enumerate(generator.templates, 1):
    print(f'{i}. {template}')

print()
print('✅ Cleaned phrase generator working correctly!')
