from unittest import TestCase
from movers import blog


class TestBlogs(TestCase):
    """Test blog fetching functions"""

    def test_blog_list_not_empty(self):
        """Test blog list is not empty."""
        blog_list = blog.blog_list()
        self.assertTrue(len(blog_list) > 0)
