from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from db.mongo import get_client, get_people_collection


class Command(BaseCommand):
    help = "Verify the MongoDB Atlas connection and face collection."

    def handle(self, *args, **options):
        try:
            client = get_client()
            client.admin.command("ping")
            collection = get_people_collection()
            count = collection.count_documents({})
        except Exception as exc:
            raise CommandError(f"MongoDB connection failed: {exc}") from exc

        self.stdout.write(self.style.SUCCESS("MongoDB ping OK"))
        self.stdout.write(f"Database: {settings.MONGODB_DB}")
        self.stdout.write(f"Collection: {collection.name} ({count} enrolled face(s))")
        self.stdout.write(f"Indexes: {list(collection.index_information())}")
