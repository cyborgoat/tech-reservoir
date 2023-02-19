from movers import blog
import dotenv

config = dotenv.dotenv_values('.env')


if __name__ == "__main__":
    blog.post_blogs(config)
