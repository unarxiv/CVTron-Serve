import unittest

class TestApp(unittest.TestCase):
    def test_run(self):
        a = 1
        self.assertIsNotNone(a)

if __name__ == '__main__':
    unittest.main()